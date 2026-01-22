from sympy.combinatorics.permutations import Permutation
from sympy.core.symbol import symbols
from sympy.matrices import Matrix
from sympy.utilities.iterables import variations, rotate_left
def rubik(n):
    """Return permutations for an nxn Rubik's cube.

    Permutations returned are for rotation of each of the slice
    from the face up to the last face for each of the 3 sides (in this order):
    front, right and bottom. Hence, the first n - 1 permutations are for the
    slices from the front.
    """
    if n < 2:
        raise ValueError('dimension of cube must be > 1')

    def getr(f, i):
        return faces[f].col(n - i)

    def getl(f, i):
        return faces[f].col(i - 1)

    def getu(f, i):
        return faces[f].row(i - 1)

    def getd(f, i):
        return faces[f].row(n - i)

    def setr(f, i, s):
        faces[f][:, n - i] = Matrix(n, 1, s)

    def setl(f, i, s):
        faces[f][:, i - 1] = Matrix(n, 1, s)

    def setu(f, i, s):
        faces[f][i - 1, :] = Matrix(1, n, s)

    def setd(f, i, s):
        faces[f][n - i, :] = Matrix(1, n, s)

    def cw(F, r=1):
        for _ in range(r):
            face = faces[F]
            rv = []
            for c in range(n):
                for r in range(n - 1, -1, -1):
                    rv.append(face[r, c])
            faces[F] = Matrix(n, n, rv)

    def ccw(F):
        cw(F, 3)

    def fcw(i, r=1):
        for _ in range(r):
            if i == 0:
                cw(F)
            i += 1
            temp = getr(L, i)
            setr(L, i, list(getu(D, i)))
            setu(D, i, list(reversed(getl(R, i))))
            setl(R, i, list(getd(U, i)))
            setd(U, i, list(reversed(temp)))
            i -= 1

    def fccw(i):
        fcw(i, 3)

    def FCW(r=1):
        for _ in range(r):
            cw(F)
            ccw(B)
            cw(U)
            t = faces[U]
            cw(L)
            faces[U] = faces[L]
            cw(D)
            faces[L] = faces[D]
            cw(R)
            faces[D] = faces[R]
            faces[R] = t

    def FCCW():
        FCW(3)

    def UCW(r=1):
        for _ in range(r):
            cw(U)
            ccw(D)
            t = faces[F]
            faces[F] = faces[R]
            faces[R] = faces[B]
            faces[B] = faces[L]
            faces[L] = t

    def UCCW():
        UCW(3)
    U, F, R, B, L, D = names = symbols('U, F, R, B, L, D')
    faces = {}
    count = 0
    for fi in range(6):
        f = []
        for a in range(n ** 2):
            f.append(count)
            count += 1
        faces[names[fi]] = Matrix(n, n, f)

    def perm(show=0):
        p = []
        for f in names:
            p.extend(faces[f])
        if show:
            return p
        g.append(Permutation(p))
    g = []
    I = list(range(6 * n ** 2))
    for i in range(n - 1):
        fcw(i)
        perm()
        fccw(i)
    assert perm(1) == I
    UCW()
    for i in range(n - 1):
        fcw(i)
        UCCW()
        perm()
        UCW()
        fccw(i)
    UCCW()
    assert perm(1) == I
    FCW()
    UCCW()
    FCCW()
    for i in range(n - 1):
        fcw(i)
        FCW()
        UCW()
        FCCW()
        perm()
        FCW()
        UCCW()
        FCCW()
        fccw(i)
    FCW()
    UCW()
    FCCW()
    assert perm(1) == I
    return g