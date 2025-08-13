import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
from NeRP_main.data import create_grid
from typing import Tuple, List
import matplotlib.pyplot as plt
import torch.nn as nn

#Forward matrix operation
def sensor_output(train_output, A_tensor):
    train_output = train_output.reshape(1024,)
    return A_tensor @ train_output

#Function to create DCT matrix (from Prof. Maleki)
def dct_matrix(N):
    D = np.zeros((N, N))
    for k in range(N):
        for n in range(N):
            if k == 0:
                alpha = np.sqrt(1 / N)
            else:
                alpha = np.sqrt(2 / N)
            D[k, n] = alpha * np.cos(np.pi * (2 * n + 1) * k / (2 * N))
    return D

#Vectorise an image
def vec(x):
    r, c = x.shape
    v = np.zeros((r*c,1))
    for i in range(r):
        for j in range(c):
            v[i*c+j] = x[i][j]
    return v

#Convert from a vector back to the image matrix
def undo_vec(v, r, c):
    y = np.zeros((r,c))
    for i in range(r):
        for j in range(c):
            y[i][j] = v[i*c+j]
    return y

#Train the model
def train_gauss(model, optim, loss_fn, data_loader, measurements, encoder, config, A_tensor=None, display=True, display_img=False, learn_from_proj=False, device="cpu", max_iter = 1000, psnr_loss=nn.MSELoss()):
    img_size = config["img_size"]
    train_psnr_list = []
    test_psnr_list = []
    for it, (grid, image) in enumerate(data_loader):
        # Input coordinates (x,y) grid and target image
        grid = grid.to(device)  # [bs, h, w, 2], [0, 1]
        image = image.to(device)  # [bs, h, w, c], [0, 1]
        # Data loading
        test_data = (grid, image)
        train_data = (grid, measurements)

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

            train_loss = 0.5 * loss_fn(train_projs, train_data[1])
            train_loss.backward()
            optim.step()

            # Compute training psnr
            if (iterations + 1) % config['log_iter'] == 0:
                psnr_train_loss = 0.5 * psnr_loss(train_projs, train_data[1])
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

                    test_loss = 0.5 * loss_fn(test_output, test_data[1])
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
                if learn_from_proj:
                    plt.title("Learned Target")
                else:
                    plt.title("Embedded Prior")
                plt.show()
    return model, train_psnr_list, test_psnr_list


#Modified from NeRP code for compatibility
class myImageDataset_2D(Dataset):

    def __init__(self, truth_img, img_dim):
        '''
        img_dim: new image size [h, w]
        '''
        self.img_dim = (img_dim, img_dim) if type(img_dim) == int else tuple(img_dim)
        self.img = torch.tensor(truth_img, dtype=torch.float32)[:, :, None]

        
    def __getitem__(self, idx):
        grid = create_grid(*self.img_dim)
        return grid, self.img

    def __len__(self):
        return 1


#From Eric
def generate_observations(dataset: np.ndarray, A: np.ndarray, sigma: float = None) -> dict:
    """
    Generates compressed observations using a sensing matrix A and adds Gaussian noise if sigma is provided.

    Parameters
    ----------
    dataset : np.ndarray
        The original image dataset where each row is an image vector
    A : np.ndarray
        The sensing matrix used for compression.
    sigma : float, optional
        The standard deviation of the Gaussian noise to be added to the compressed observations.
        If not provided, no noise will be added.

    Returns
    -------
    dict
        A dictionary containing the compressed observations "X" and the original dataset "Y".
    """

    n, _ = dataset.shape
    m, _ = A.shape

    X = np.zeros((n, m))
    Y = dataset / 255   

    for i, x in enumerate(Y):

        y = A @ x
        
        if sigma is not None:

            rng = np.random.default_rng(i)  # Create a new RNG with a seed based on the image index
            noise = sigma * rng.normal(0, 1, size=(m,))
            X[i] = y + noise

        else:

            X[i] = y

    return {"X": X, "Y": Y}
def gaussian_matrix(shape: Tuple[int, int], seed: int = None) -> np.ndarray:
    """
    Constructs a Gaussian sensing matrix A where each element A_ij is drawn from a Gaussian distribution N(0, 1/m).

    Parameters
    ----------
    shape : Tuple[int, int]
        The shape of the matrix A as (rows, columns).
    seed : int, optional
        The seed value for the internal NumPy random number generator. If not provided, the RNG will be initialized
        with a random seed value.

    Returns
    -------
    np.ndarray
        The constructed Gaussian sensing matrix A.
    """

    if shape[0] > shape[1]:
        raise ValueError("Row count greater than column count")
    
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1 / np.sqrt(shape[0]), shape)

def normalize_max_row_norm(matrix: np.ndarray) -> np.ndarray:
    """
    Normalizes the rows of the input matrix by the maximum norm.

    Parameters
    ----------
    matrix : np.ndarray
        The input matrix to be normalized. It should be a 2D array.

    Returns
    -------
    np.ndarray
        The matrix with rows normalized by the maximum norm of the rows.
    """
    row_norms = np.linalg.norm(matrix, axis=1)  
    max_norm = np.max(row_norms)                 
    normalized_matrix = matrix / max_norm        

    return normalized_matrix