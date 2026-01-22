import numpy
import cupy
import cupyx.cusolver
from cupy import cublas
from cupyx import cusparse
from cupy.cuda import cusolver
from cupy.cuda import device
from cupy.cuda import runtime
from cupy.linalg import _util
from cupy_backends.cuda.libs import cusparse as _cusparse
from cupyx.scipy import sparse
from cupyx.scipy.sparse.linalg import _interface
from cupyx.scipy.sparse.linalg._iterative import _make_system
import warnings
def lsmr(A, b, x0=None, damp=0.0, atol=1e-06, btol=1e-06, conlim=100000000.0, maxiter=None):
    """Iterative solver for least-squares problems.

    lsmr solves the system of linear equations ``Ax = b``. If the system
    is inconsistent, it solves the least-squares problem ``min ||b - Ax||_2``.
    A is a rectangular matrix of dimension m-by-n, where all cases are
    allowed: m = n, m > n, or m < n. B is a vector of length m.
    The matrix A may be dense or sparse (usually sparse).

    Args:
        A (ndarray, spmatrix or LinearOperator): The real or complex
            matrix of the linear system. ``A`` must be
            :class:`cupy.ndarray`, :class:`cupyx.scipy.sparse.spmatrix` or
            :class:`cupyx.scipy.sparse.linalg.LinearOperator`.
        b (cupy.ndarray): Right hand side of the linear system with shape
            ``(m,)`` or ``(m, 1)``.
        x0 (cupy.ndarray): Starting guess for the solution. If None zeros are
            used.
        damp (float): Damping factor for regularized least-squares.
            `lsmr` solves the regularized least-squares problem
            ::

                min ||(b) - (  A   )x||
                    ||(0)   (damp*I) ||_2

            where damp is a scalar. If damp is None or 0, the system
            is solved without regularization.
        atol, btol (float):
            Stopping tolerances. `lsmr` continues iterations until a
            certain backward error estimate is smaller than some quantity
            depending on atol and btol.
        conlim (float): `lsmr` terminates if an estimate of ``cond(A)`` i.e.
            condition number of matrix exceeds `conlim`. If `conlim` is None,
            the default value is 1e+8.
        maxiter (int): Maximum number of iterations.

    Returns:
        tuple:
            - `x` (ndarray): Least-square solution returned.
            - `istop` (int): istop gives the reason for stopping::

                    0 means x=0 is a solution.

                    1 means x is an approximate solution to A*x = B,
                    according to atol and btol.

                    2 means x approximately solves the least-squares problem
                    according to atol.

                    3 means COND(A) seems to be greater than CONLIM.

                    4 is the same as 1 with atol = btol = eps (machine
                    precision)

                    5 is the same as 2 with atol = eps.

                    6 is the same as 3 with CONLIM = 1/eps.

                    7 means ITN reached maxiter before the other stopping
                    conditions were satisfied.

            - `itn` (int): Number of iterations used.
            - `normr` (float): ``norm(b-Ax)``
            - `normar` (float): ``norm(A^T (b - Ax))``
            - `norma` (float): ``norm(A)``
            - `conda` (float): Condition number of A.
            - `normx` (float): ``norm(x)``

    .. seealso:: :func:`scipy.sparse.linalg.lsmr`

    References:
        D. C.-L. Fong and M. A. Saunders, "LSMR: An iterative algorithm for
        sparse least-squares problems", SIAM J. Sci. Comput.,
        vol. 33, pp. 2950-2971, 2011.
    """
    A = _interface.aslinearoperator(A)
    b = b.squeeze()
    matvec = A.matvec
    rmatvec = A.rmatvec
    m, n = A.shape
    minDim = min([m, n])
    if maxiter is None:
        maxiter = minDim * 5
    u = b.copy()
    normb = cublas.nrm2(b)
    beta = normb.copy()
    normb = normb.get().item()
    if x0 is None:
        x = cupy.zeros((n,), dtype=A.dtype)
    else:
        if not (x0.shape == (n,) or x0.shape == (n, 1)):
            raise ValueError('x0 has incompatible dimensions')
        x = x0.astype(A.dtype).ravel()
        u -= matvec(x)
        beta = cublas.nrm2(u)
    beta_cpu = beta.get().item()
    v = cupy.zeros(n)
    alpha = cupy.zeros((), dtype=beta.dtype)
    alpha_cpu = 0
    if beta_cpu > 0:
        u /= beta
        v = rmatvec(u)
        alpha = cublas.nrm2(v)
        alpha_cpu = alpha.get().item()
    if alpha_cpu > 0:
        v /= alpha
    itn = 0
    zetabar = alpha_cpu * beta_cpu
    alphabar = alpha_cpu
    rho = 1
    rhobar = 1
    cbar = 1
    sbar = 0
    h = v.copy()
    hbar = cupy.zeros(n)
    betadd = beta_cpu
    betad = 0
    rhodold = 1
    tautildeold = 0
    thetatilde = 0
    zeta = 0
    d = 0
    normA2 = alpha_cpu * alpha_cpu
    maxrbar = 0
    minrbar = 1e+100
    normA = alpha_cpu
    condA = 1
    normx = 0
    istop = 0
    ctol = 0
    if conlim > 0:
        ctol = 1 / conlim
    normr = beta_cpu
    normar = alpha_cpu * beta_cpu
    if normar == 0:
        return (x, istop, itn, normr, normar, normA, condA, normx)
    while itn < maxiter:
        itn = itn + 1
        u *= -alpha
        u += matvec(v)
        beta = cublas.nrm2(u)
        beta_cpu = beta.get().item()
        if beta_cpu > 0:
            u /= beta
            v *= -beta
            v += rmatvec(u)
            alpha = cublas.nrm2(v)
            alpha_cpu = alpha.get().item()
            if alpha_cpu > 0:
                v /= alpha
        chat, shat, alphahat = _symOrtho(alphabar, damp)
        rhoold = rho
        c, s, rho = _symOrtho(alphahat, beta_cpu)
        thetanew = s * alpha_cpu
        alphabar = c * alpha_cpu
        rhobarold = rhobar
        zetaold = zeta
        thetabar = sbar * rho
        rhotemp = cbar * rho
        cbar, sbar, rhobar = _symOrtho(cbar * rho, thetanew)
        zeta = cbar * zetabar
        zetabar = -sbar * zetabar
        hbar *= -(thetabar * rho / (rhoold * rhobarold))
        hbar += h
        x += zeta / (rho * rhobar) * hbar
        h *= -(thetanew / rho)
        h += v
        betaacute = chat * betadd
        betacheck = -shat * betadd
        betahat = c * betaacute
        betadd = -s * betaacute
        thetatildeold = thetatilde
        ctildeold, stildeold, rhotildeold = _symOrtho(rhodold, thetabar)
        thetatilde = stildeold * rhobar
        rhodold = ctildeold * rhobar
        betad = -stildeold * betad + ctildeold * betahat
        tautildeold = (zetaold - thetatildeold * tautildeold) / rhotildeold
        taud = (zeta - thetatilde * tautildeold) / rhodold
        d = d + betacheck * betacheck
        normr = numpy.sqrt(d + (betad - taud) ** 2 + betadd * betadd)
        normA2 = normA2 + beta_cpu * beta_cpu
        normA = numpy.sqrt(normA2)
        normA2 = normA2 + alpha_cpu * alpha_cpu
        maxrbar = max(maxrbar, rhobarold)
        if itn > 1:
            minrbar = min(minrbar, rhobarold)
        condA = max(maxrbar, rhotemp) / min(minrbar, rhotemp)
        normar = abs(zetabar)
        normx = cublas.nrm2(x)
        normx = normx.get().item()
        test1 = normr / normb
        if normA * normr != 0:
            test2 = normar / (normA * normr)
        else:
            test2 = numpy.infty
        test3 = 1 / condA
        t1 = test1 / (1 + normA * normx / normb)
        rtol = btol + atol * normA * normx / normb
        if itn >= maxiter:
            istop = 7
        if 1 + test3 <= 1:
            istop = 6
        if 1 + test2 <= 1:
            istop = 5
        if 1 + t1 <= 1:
            istop = 4
        if test3 <= ctol:
            istop = 3
        if test2 <= atol:
            istop = 2
        if test1 <= rtol:
            istop = 1
        if istop > 0:
            break
    x = x.astype(numpy.float64)
    return (x, istop, itn, normr, normar, normA, condA, normx)