import numpy as np
def fg1eu(x):
    """Eubank similar to Fan and Gijbels example function 1

    """
    return x + 0.5 * np.exp(-50 * (x - 0.5) ** 2)