import warnings
from scipy.linalg import sqrtm
import numpy as np
import pennylane as qml
Post process the corresponding tape results to get the metric tensor estimation.

        Args:
            tensor_raw_results list[np.array]: list of the four perturbed qnode results to compute
            the estimated metric tensor
            tensor_dirs list[np.array]: list of the two perturbation directions used to compute
            the metric tensor estimation. Perturbations on the different input parameters have
            been concatenated

        Returns:
            np.array: estimated Fubini-Study metric tensor
        