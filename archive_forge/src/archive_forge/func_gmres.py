import numpy
import cupy
from cupy import cublas
from cupyx import cusparse
from cupy._core import _dtype
from cupy.cuda import device
from cupy_backends.cuda.libs import cusparse as _cusparse
from cupy_backends.cuda.libs import cublas as _cublas
from cupyx.scipy.sparse import _csr
from cupyx.scipy.sparse.linalg import _interface
def gmres(A, b, x0=None, tol=1e-05, restart=None, maxiter=None, M=None, callback=None, atol=None, callback_type=None):
    """Uses Generalized Minimal RESidual iteration to solve ``Ax = b``.

    Args:
        A (ndarray, spmatrix or LinearOperator): The real or complex
            matrix of the linear system with shape ``(n, n)``. ``A`` must be
            :class:`cupy.ndarray`, :class:`cupyx.scipy.sparse.spmatrix` or
            :class:`cupyx.scipy.sparse.linalg.LinearOperator`.
        b (cupy.ndarray): Right hand side of the linear system with shape
            ``(n,)`` or ``(n, 1)``.
        x0 (cupy.ndarray): Starting guess for the solution.
        tol (float): Tolerance for convergence.
        restart (int): Number of iterations between restarts. Larger values
            increase iteration cost, but may be necessary for convergence.
        maxiter (int): Maximum number of iterations.
        M (ndarray, spmatrix or LinearOperator): Preconditioner for ``A``.
            The preconditioner should approximate the inverse of ``A``.
            ``M`` must be :class:`cupy.ndarray`,
            :class:`cupyx.scipy.sparse.spmatrix` or
            :class:`cupyx.scipy.sparse.linalg.LinearOperator`.
        callback (function): User-specified function to call on every restart.
            It is called as ``callback(arg)``, where ``arg`` is selected by
            ``callback_type``.
        callback_type (str): 'x' or 'pr_norm'. If 'x', the current solution
            vector is used as an argument of callback function. if 'pr_norm',
            relative (preconditioned) residual norm is used as an arugment.
        atol (float): Tolerance for convergence.

    Returns:
        tuple:
            It returns ``x`` (cupy.ndarray) and ``info`` (int) where ``x`` is
            the converged solution and ``info`` provides convergence
            information.

    Reference:
        M. Wang, H. Klie, M. Parashar and H. Sudan, "Solving Sparse Linear
        Systems on NVIDIA Tesla GPUs", ICCS 2009 (2009).

    .. seealso:: :func:`scipy.sparse.linalg.gmres`
    """
    A, M, x, b = _make_system(A, M, x0, b)
    matvec = A.matvec
    psolve = M.matvec
    n = A.shape[0]
    if n == 0:
        return (cupy.empty_like(b), 0)
    b_norm = cupy.linalg.norm(b)
    if b_norm == 0:
        return (b, 0)
    if atol is None:
        atol = tol * float(b_norm)
    else:
        atol = max(float(atol), tol * float(b_norm))
    if maxiter is None:
        maxiter = n * 10
    if restart is None:
        restart = 20
    restart = min(restart, n)
    if callback_type is None:
        callback_type = 'pr_norm'
    if callback_type not in ('x', 'pr_norm'):
        raise ValueError('Unknown callback_type: {}'.format(callback_type))
    if callback is None:
        callback_type = None
    V = cupy.empty((n, restart), dtype=A.dtype, order='F')
    H = cupy.zeros((restart + 1, restart), dtype=A.dtype, order='F')
    e = numpy.zeros((restart + 1,), dtype=A.dtype)
    compute_hu = _make_compute_hu(V)
    iters = 0
    while True:
        mx = psolve(x)
        r = b - matvec(mx)
        r_norm = cublas.nrm2(r)
        if callback_type == 'x':
            callback(mx)
        elif callback_type == 'pr_norm' and iters > 0:
            callback(r_norm / b_norm)
        if r_norm <= atol or iters >= maxiter:
            break
        v = r / r_norm
        V[:, 0] = v
        e[0] = r_norm
        for j in range(restart):
            z = psolve(v)
            u = matvec(z)
            H[:j + 1, j], u = compute_hu(u, j)
            cublas.nrm2(u, out=H[j + 1, j])
            if j + 1 < restart:
                v = u / H[j + 1, j]
                V[:, j + 1] = v
        ret = numpy.linalg.lstsq(cupy.asnumpy(H), e)
        y = cupy.array(ret[0])
        x += V @ y
        iters += restart
    info = 0
    if iters == maxiter and (not r_norm <= atol):
        info = iters
    return (mx, info)