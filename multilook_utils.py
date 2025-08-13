from my_utils import *
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
from typing import Tuple, List
import matplotlib.pyplot as plt
import torch.nn as nn


#Train the model
def train_multilook_gauss(model, optim, loss_fn, data_loader, measurements, encoder, config, A_tensor=None, display=True, display_img=False, learn_from_proj=False, device="cpu", max_iter = 1000, psnr_loss=nn.MSELoss()):
    img_size = config["img_size"]
    num_looks = config["num_looks"]
    train_psnr_list = []
    test_psnr_list = []
    for it, (grid, image) in enumerate(data_loader):
        # Input coordinates (x,y) grid and target image
        grid = grid.to(device)  # [bs, h, w, 2], [0, 1]
        image = image.to(device)  # [bs, h, w, c], [0, 1]
        # print(grid.shape, image.shape)
        # Data loading
        test_data = (grid, image)
        train_data = (grid, measurements)
    
        #Get initial PSNR values before the model is trained
        with torch.no_grad():
            test_embedding = encoder.embedding(test_data[0])
            test_output = model(test_embedding)

            test_loss = 0.5 * loss_fn(test_output, test_data[1], 1)
            psnr_test_loss = 0.5 * psnr_loss(test_output, test_data[1])
            test_psnr = - 10 * torch.log10(2 * psnr_test_loss).item()
            test_loss = test_loss.item()
            test_psnr_list.append(test_psnr)

            train_embedding = encoder.embedding(train_data[0]) 
            train_output = model(train_embedding)
            if learn_from_proj:
                train_projs = sensor_output(train_output, A_tensor)
            else:
                train_projs = train_output

            psnr_train_loss = 0.5 * psnr_loss(train_projs, train_data[1][0])
            train_psnr = -10 * torch.log10(2 * psnr_train_loss).item()
            train_psnr_list.append(train_psnr)

        # Train model
        for iterations in range(max_iter):
            model.train()
            optim.zero_grad()

            train_embedding = encoder.embedding(train_data[0])  # [B, H, W, embedding*2]
            train_output = model(train_embedding)  # [B, H, W, 3]

            if learn_from_proj:
                train_projs = sensor_output(train_output, A_tensor)
            else:
                train_projs = train_output

            train_loss = 0.5 * loss_fn(train_projs, train_data[1], num_looks)
            train_loss.backward()
            optim.step()

            # Compute training psnr
            if (iterations + 1) % config['log_iter'] == 0:
                psnr_train_loss = 0.5 * psnr_loss(train_projs, train_data[1][0])
                train_psnr = -10 * torch.log10(2 * psnr_train_loss).item()
                train_loss = train_loss.item()
                if(display or iterations == max_iter-1):
                    print("[Iteration: {}/{}] Train loss: {:.4g} | Train psnr: {:.4g}".format(iterations + 1, max_iter, train_loss, train_psnr))
                train_psnr_list.append(train_psnr)
            # Compute testing psnr
            if iterations == 0 or (iterations + 1) % config['val_iter'] == 0:
                model.eval()
                with torch.no_grad():
                    test_embedding = encoder.embedding(test_data[0])
                    test_output = model(test_embedding)

                    test_loss = 0.5 * loss_fn(test_output, test_data[1], 1)
                    psnr_test_loss = 0.5 * psnr_loss(test_output, test_data[1])
                    test_psnr = - 10 * torch.log10(2 * psnr_test_loss).item()
                    test_loss = test_loss.item()
                    test_psnr_list.append(test_psnr)
                if(display or iterations == max_iter-1):
                    print("[Validation Iteration: {}/{}] Test loss: {:.4g} | Test psnr: {:.4g}".format(iterations + 1, max_iter, test_loss, test_psnr))
                if display_img:
                    output = train_output.to('cpu').detach().numpy().reshape((img_size,img_size))
                    plt.imshow(output, cmap='gray')
                    plt.show()
            if iterations == max_iter-1:
                output = train_output.to('cpu').detach().numpy().reshape((img_size,img_size))
                plt.imshow(output, cmap='gray')
                plt.title("Learned Target")
                plt.show()
    return model, train_psnr_list, test_psnr_list