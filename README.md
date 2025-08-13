# Sarah-Zabelle_NeRP_Based_Speckle_Noise_Research
The code from our summer 2025 research project based off NeRP.
We applied this idea to both Gaussian measurement matrices and the speckle noise case.

The folder NeRP_main contains the code from the NeRP paper:
Shen, Liyue, Pauly, John, and Xing, Lei.  
**NeRP: Implicit Neural Representation Learning with Prior Embedding for Sparsely Sampled Image Reconstruction**.  
*IEEE Transactions on Neural Networks and Learning Systems*, 2022.  
[IEEE](https://ieeexplore.ieee.org/document/9788018) 

Coherent_Imaging.ipynb contains code to generate the speckle noise datasets

NeRP_for_gaussians.ipynb contains code to run INR with a prior on measurements from a gaussian sampling matrix A
- The corresponding config file is gauss_recon.yaml

INR_speckle_multilook.ipynb contains code to run INR with a prior for the speckle noise and multilook case
- The corresponding config file is speckle_recon.yaml

my_utils.py and multilook_utils.py contain code based off of the NeRP code to train the models as well as some additional helper functions such as generating the gaussian observations written by Eric Chen
