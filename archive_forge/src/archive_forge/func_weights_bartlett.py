import numpy as np
def weights_bartlett(nlags):
    return 1 - np.arange(nlags + 1) / (nlags + 1.0)