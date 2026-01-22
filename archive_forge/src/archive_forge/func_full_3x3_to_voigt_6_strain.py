import numpy as np
def full_3x3_to_voigt_6_strain(strain_matrix):
    """
    Form a 6 component strain vector in Voigt notation from a 3x3 matrix
    """
    strain_matrix = np.asarray(strain_matrix)
    return np.transpose([strain_matrix[..., 0, 0] - 1.0, strain_matrix[..., 1, 1] - 1.0, strain_matrix[..., 2, 2] - 1.0, strain_matrix[..., 1, 2] + strain_matrix[..., 2, 1], strain_matrix[..., 0, 2] + strain_matrix[..., 2, 0], strain_matrix[..., 0, 1] + strain_matrix[..., 1, 0]])