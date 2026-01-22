import math
from cvxopt import base, blas, lapack, cholmod, misc_solvers
from cvxopt.base import matrix, spmatrix
def kkt_ldl2(G, dims, A, mnl=0):
    """
    Solution of KKT equations by a dense LDL factorization of the 2 x 2 
    system.
    
    Returns a function that (1) computes the LDL factorization of
    
        [ H + GG' * W^{-1} * W^{-T} * GG   A' ]
        [                                     ]
        [ A                                0  ]
    
    given H, Df, W, where GG = [Df; G], and (2) returns a function for 
    solving 
    
        [ H    A'   GG'   ]   [ ux ]   [ bx ]
        [ A    0    0     ] * [ uy ] = [ by ].
        [ GG   0   -W'*W  ]   [ uz ]   [ bz ]
    
    H is n x n,  A is p x n, Df is mnl x n, G is N x n where
    N = dims['l'] + sum(dims['q']) + sum( k**2 for k in dims['s'] ).
    """
    p, n = A.size
    ldK = n + p
    K = matrix(0.0, (ldK, ldK))
    if p:
        ipiv = matrix(0, (ldK, 1))
    g = matrix(0.0, (mnl + G.size[0], 1))
    u = matrix(0.0, (ldK, 1))

    def factor(W, H=None, Df=None):
        blas.scal(0.0, K)
        if H is not None:
            K[:n, :n] = H
        K[n:, :n] = A
        for k in range(n):
            if mnl:
                g[:mnl] = Df[:, k]
            g[mnl:] = G[:, k]
            scale(g, W, trans='T', inverse='I')
            scale(g, W, inverse='I')
            if mnl:
                base.gemv(Df, g, K, trans='T', beta=1.0, n=n - k, offsetA=mnl * k, offsety=(ldK + 1) * k)
            sgemv(G, g, K, dims, trans='T', beta=1.0, n=n - k, offsetA=G.size[0] * k, offsetx=mnl, offsety=(ldK + 1) * k)
        if p:
            lapack.sytrf(K, ipiv)
        else:
            lapack.potrf(K)

        def solve(x, y, z):
            blas.copy(z, g)
            scale(g, W, trans='T', inverse='I')
            scale(g, W, inverse='I')
            if mnl:
                base.gemv(Df, g, u, trans='T')
                beta = 1.0
            else:
                beta = 0.0
            sgemv(G, g, u, dims, trans='T', offsetx=mnl, beta=beta)
            blas.axpy(x, u)
            blas.copy(y, u, offsety=n)
            if p:
                lapack.sytrs(K, ipiv, u)
            else:
                lapack.potrs(K, u)
            blas.copy(u, x, n=n)
            blas.copy(u, y, offsetx=n, n=p)
            if mnl:
                base.gemv(Df, x, z, alpha=1.0, beta=-1.0)
            sgemv(G, x, z, dims, alpha=1.0, beta=-1.0, offsety=mnl)
            scale(z, W, trans='T', inverse='I')
        return solve
    return factor