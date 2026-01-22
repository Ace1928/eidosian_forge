import numpy as np
def full_3x3_to_voigt_6_stress(stress_matrix):
    """
    Form a 6 component stress vector in Voigt notation from a 3x3 matrix
    """
    stress_matrix = np.asarray(stress_matrix)
    return np.transpose([stress_matrix[..., 0, 0], stress_matrix[..., 1, 1], stress_matrix[..., 2, 2], (stress_matrix[..., 1, 2] + stress_matrix[..., 1, 2]) / 2, (stress_matrix[..., 0, 2] + stress_matrix[..., 0, 2]) / 2, (stress_matrix[..., 0, 1] + stress_matrix[..., 0, 1]) / 2])