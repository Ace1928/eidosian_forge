import numpy as np
def voigt_6_to_full_3x3_strain(strain_vector):
    """
    Form a 3x3 strain matrix from a 6 component vector in Voigt notation
    """
    e1, e2, e3, e4, e5, e6 = np.transpose(strain_vector)
    return np.transpose([[1.0 + e1, 0.5 * e6, 0.5 * e5], [0.5 * e6, 1.0 + e2, 0.5 * e4], [0.5 * e5, 0.5 * e4, 1.0 + e3]])