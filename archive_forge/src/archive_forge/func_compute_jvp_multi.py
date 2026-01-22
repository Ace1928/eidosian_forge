import numpy as np
import pennylane as qml
from pennylane.measurements import ProbabilityMP
def compute_jvp_multi(tangent, jac):
    """Convenience function to compute the Jacobian-vector product for a given
    vector of gradient outputs and a Jacobian for a tape with multiple measurements.

    Args:
        tangent (tensor_like, list): tangent vector
        jac (tensor_like, tuple): Jacobian matrix

    Returns:
        tensor_like: the Jacobian-vector product

    **Examples**

    1. For a single parameter and multiple measurements (one without shape and one with shape, e.g. expval and probs):

    .. code-block:: pycon

        >>> tangent = np.array([2.0])
        >>> jac = tuple([np.array([0.3]), np.array([0.2, 0.5])])
        >>> qml.gradients.compute_jvp_multi(tangent, jac)
        (np.array([0.6]), np.array([0.4, 1. ]))

    2. For multiple parameters (in this case 2 parameters) and multiple measurements (one without shape and one with
    shape, e.g. expval and probs):

    .. code-block:: pycon

        >>> tangent = np.array([1.0, 2.0])
        >>> jac = tuple([tuple([np.array([0.3]), np.array([0.4])]), tuple([np.array([0.2, 0.5]), np.array([0.3, 0.8])]),])
        >>> qml.gradients.compute_jvp_multi(tangent, jac)
        (np.array([1.1]), np.array([0.8, 2.1]))
    """
    if jac is None:
        return None
    return tuple((compute_jvp_single(tangent, j) for j in jac))