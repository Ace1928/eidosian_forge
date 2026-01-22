import numpy as np
def closest_psd_matrix(K, fix_diagonal=False, solver=None, **kwargs):
    """Return the closest positive semi-definite matrix to the given kernel matrix.

    This method either fixes the diagonal entries to be 1
    (``fix_diagonal=True``) or keeps the eigenvectors intact (``fix_diagonal=False``),
    in which case it reduces to the method :func:`~.kernels.threshold_matrix`.
    For ``fix_diagonal=True`` a semi-definite program is solved.

    Args:
        K (array[float]): Kernel matrix, assumed to be symmetric.
        fix_diagonal (bool): Whether to fix the diagonal to 1.
        solver (str, optional): Solver to be used by cvxpy. Defaults to CVXOPT.
        kwargs (kwarg dict): Passed to cvxpy.Problem.solve().

    Returns:
        array[float]: closest positive semi-definite matrix in Frobenius norm.

    Comments:
        Requires cvxpy and the used solver (default CVXOPT) to be installed if ``fix_diagonal=True``.

    Reference:
        This method is introduced in `arXiv:2105.02276 <https://arxiv.org/abs/2105.02276>`_.

    **Example:**

    Consider a symmetric matrix with both positive and negative eigenvalues:

    .. code-block :: pycon

        >>> K = np.array([[0.9, 1.], [1., 0.9]])
        >>> np.linalg.eigvalsh(K)
        array([-0.1, 1.9])

    The positive semi-definite matrix that is closest to this matrix in any unitarily
    invariant norm is then given by the matrix with the eigenvalues thresholded at 0,
    as computed by :func:`~.kernels.threshold_matrix`:

    .. code-block :: pycon

        >>> K_psd = qml.kernels.closest_psd_matrix(K)
        >>> K_psd
        tensor([[0.95, 0.95],
                [0.95, 0.95]], requires_grad=True)
        >>> np.linalg.eigvalsh(K_psd)
        array([0., 1.9])
        >>> np.allclose(K_psd, qml.kernels.threshold_matrix(K))
        True

    However, for quantum kernel matrices we may want to restore the value 1 on the
    diagonal:

    .. code-block :: pycon

        >>> K_psd = qml.kernels.closest_psd_matrix(K, fix_diagonal=True)
        >>> K_psd
        array([[1.        , 0.99998008],
               [0.99998008, 1.        ]])
        >>> np.linalg.eigvalsh(K_psd)
        array([1.99162415e-05, 1.99998008e+00])

    If the input matrix does not have negative eigenvalues and ``fix_diagonal=False``,
    ``closest_psd_matrix`` does not have any effect.
    """
    if not fix_diagonal:
        return threshold_matrix(K)
    try:
        import cvxpy as cp
        if solver is None:
            solver = cp.CVXOPT
    except ImportError as e:
        raise ImportError('CVXPY is required for this post-processing method.') from e
    X = cp.Variable(K.shape, PSD=True)
    constraint = [cp.diag(X) == 1.0] if fix_diagonal else []
    objective_fn = cp.norm(X - K, 'fro')
    problem = cp.Problem(cp.Minimize(objective_fn), constraint)
    try:
        problem.solve(solver=solver, **kwargs)
    except Exception:
        try:
            problem.solve(solver=solver, verbose=True, **kwargs)
        except Exception as e:
            raise RuntimeError('CVXPY solver did not converge.') from e
    return X.value