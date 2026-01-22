import sys
def kktsolver_e(x, znl, W):
    We = W.copy()
    We['dnl'] = W['dnl'][1:]
    We['dnli'] = W['dnli'][1:]
    g = kktsolver(x[0], znl, We)
    f, Df = F(x[0])
    if type(Df) is matrix:
        gradf0 = Df[0, :].T
    elif type(Df) is spmatrix:
        gradf0 = matrix(Df[0, :].T)
    else:
        gradf0 = xnewcopy(x[0])
        e0 = matrix(0.0, (mnl + 1, 1))
        e0[0] = 1.0
        Df(e0, gradf0, trans='T')

    def solve(x, y, z):
        a = z[0]
        xcopy(x[0], ux)
        xaxpy(gradf0, ux, alpha=x[1])
        blas.copy(z, uz, offsetx=1)
        g(ux, y, uz)
        z[0] = -x[1] * W['dnl'][0]
        blas.copy(uz, z, offsety=1)
        xcopy(ux, x[0])
        x[1] = xdot(gradf0, x[0]) + W['dnl'][0] ** 2 * x[1] - a
    return solve