import numpy as np
def threshold_matrix(K):
    """Remove negative eigenvalues from the given kernel matrix.

    This method yields the closest positive semi-definite matrix in
    any unitarily invariant norm, e.g. the Frobenius norm.

    Args:
        K (array[float]): Kernel matrix, assumed to be symmetric.

    Returns:
        array[float]: Kernel matrix with cropped negative eigenvalues.

    **Example:**

    Consider a symmetric matrix with both positive and negative eigenvalues:

    .. code-block :: pycon

        >>> K = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 2]])
        >>> np.linalg.eigvalsh(K)
        array([-1.,  1.,  2.])

    We then can threshold/truncate the eigenvalues of the matrix via

    .. code-block :: pycon

        >>> K_thresh = qml.kernels.threshold_matrix(K)
        >>> np.linalg.eigvalsh(K_thresh)
        array([0.,  1.,  2.])

    If the input matrix does not have negative eigenvalues, ``threshold_matrix``
    does not have any effect.
    """
    w, v = np.linalg.eigh(K)
    if w[0] < 0:
        w0 = np.clip(w, 0, None)
        return v * w0 @ np.transpose(v)
    return K