#加噪模块
import numpy as np
def pnoise(data, nsr, seed=10):
    # Add random noise to the input data
    # Inputs:
    #   data: original data
    #   nsr: noise-to-signal ratio
    #   seed: seed for random number generation
    # Outputs:
    #   datans: data with added noise
    
    if nsr == 0:
        return data
    
    np.random.seed(seed)
    s_ener = np.linalg.norm(data)**2 # Signal energy
    zao = np.random.randn(*data.shape) # Generate Gaussian noise
    zao_ener = np.linalg.norm(zao)**2 # Noise energy
    factor = np.sqrt((s_ener / zao_ener) * nsr)
    noise = factor * zao
    datans = data + noise
    return datans

