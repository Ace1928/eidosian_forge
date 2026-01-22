import numpy as np
def fg1(x):
    """Fan and Gijbels example function 1

    """
    return x + 2 * np.exp(-16 * x ** 2)