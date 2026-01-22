import math
from cvxopt import base, blas, lapack, cholmod, misc_solvers
from cvxopt.base import matrix, spmatrix
def kkt_chol(G, dims, A, mnl=0):
    """
    Solution of KKT equations by reduction to a 2 x 2 system, a QR 
    factorization to eliminate the equality constraints, and a dense 
    Cholesky factorization of order n-p. 
    
    Computes the QR factorization
    
        A' = [Q1, Q2] * [R; 0]
    
    and returns a function that (1) computes the Cholesky factorization 
    
        Q_2^T * (H + GG^T * W^{-1} * W^{-T} * GG) * Q2 = L * L^T, 
    
    given H, Df, W, where GG = [Df; G], and (2) returns a function for 
    solving 
    
        [ H    A'   GG'    ]   [ ux ]   [ bx ]
        [ A    0    0      ] * [ uy ] = [ by ].
        [ GG   0    -W'*W  ]   [ uz ]   [ bz ]
    
    H is n x n,  A is p x n, Df is mnl x n, G is N x n where
    N = dims['l'] + sum(dims['q']) + sum( k**2 for k in dims['s'] ).
    """
    p, n = A.size
    cdim = mnl + dims['l'] + sum(dims['q']) + sum([k ** 2 for k in dims['s']])
    cdim_pckd = mnl + dims['l'] + sum(dims['q']) + sum([int(k * (k + 1) / 2) for k in dims['s']])
    if type(A) is matrix:
        QA = A.T
    else:
        QA = matrix(A.T)
    tauA = matrix(0.0, (p, 1))
    lapack.geqrf(QA, tauA)
    Gs = matrix(0.0, (cdim, n))
    K = matrix(0.0, (n, n))
    bzp = matrix(0.0, (cdim_pckd, 1))
    yy = matrix(0.0, (p, 1))

    def factor(W, H=None, Df=None):
        if mnl:
            Gs[:mnl, :] = Df
        Gs[mnl:, :] = G
        scale(Gs, W, trans='T', inverse='I')
        pack2(Gs, dims, mnl)
        blas.syrk(Gs, K, k=cdim_pckd, trans='T')
        if H is not None:
            K[:, :] += H
        symm(K, n)
        lapack.ormqr(QA, tauA, K, side='L', trans='T')
        lapack.ormqr(QA, tauA, K, side='R')
        lapack.potrf(K, n=n - p, offsetA=p * (n + 1))

        def solve(x, y, z):
            scale(z, W, trans='T', inverse='I')
            pack(z, bzp, dims, mnl)
            blas.gemv(Gs, bzp, x, beta=1.0, trans='T', m=cdim_pckd)
            lapack.ormqr(QA, tauA, x, side='L', trans='T')
            blas.copy(y, yy)
            blas.copy(x, y, n=p)
            blas.copy(yy, x)
            lapack.trtrs(QA, x, uplo='U', trans='T', n=p)
            blas.gemv(K, x, x, alpha=-1.0, beta=1.0, m=n - p, n=p, offsetA=p, offsety=p)
            lapack.potrs(K, x, n=n - p, offsetA=p * (n + 1), offsetB=p)
            blas.gemv(K, x, y, alpha=-1.0, beta=1.0, m=p, n=n)
            lapack.trtrs(QA, y, uplo='U', n=p)
            lapack.ormqr(QA, tauA, x, side='L')
            blas.gemv(Gs, x, bzp, alpha=1.0, beta=-1.0, m=cdim_pckd)
            unpack(bzp, z, dims, mnl)
        return solve
    return factor