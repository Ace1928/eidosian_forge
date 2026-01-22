from ... import sage_helper
from .. import t3mlite as t3m
from . import link, dual_cellulation
def peripheral_curve_package(snappy_manifold):
    """
    Given a 1-cusped snappy_manifold M, this function returns

    1. A t3m MComplex of M, and

    2. the induced cusp triangulation, and

    3. the dual to the cusp triangulation, and

    4. two 1-cocycles on the dual cellulation which are
    *algebraically* dual to the peripheral framing of M.

    sage: M = peripheral_curve_package(Manifold('t00000'))[0]
    sage: len(M)
    8
    sage: T, D = M.cusp_triangulation, M.cusp_dual_cellulation
    sage: T.homology_test()
    sage: D.euler()
    0
    sage: D.slope(D.meridian)
    (1, 0)
    sage: D.slope(D.longitude)
    (0, 1)
    """
    M = snappy_manifold
    assert M.num_cusps() == 1
    N = t3m.Mcomplex(M)
    C = link.LinkSurface(N)
    D = dual_cellulation.DualCellulation(C)
    cusp_indices, data = M._get_cusp_indices_and_peripheral_curve_data()
    meridian = peripheral_curve_from_snappy(D, [data[i] for i in range(0, len(data), 4)])
    longitude = peripheral_curve_from_snappy(D, [data[i] for i in range(2, len(data), 4)])
    alpha, beta = D.integral_cohomology_basis()
    A = matrix([[alpha(meridian), beta(meridian)], [alpha(longitude), beta(longitude)]])
    assert abs(A.det()) == 1
    Ainv = A.inverse().change_ring(ZZ)
    B = Ainv.transpose() * matrix(ZZ, [alpha.weights, beta.weights])
    mstar = dual_cellulation.OneCocycle(D, list(B[0]))
    lstar = dual_cellulation.OneCocycle(D, list(B[1]))
    AA = matrix([[mstar(meridian), lstar(meridian)], [mstar(longitude), lstar(longitude)]])
    assert AA == 1
    N.cusp_triangulation = C
    N.cusp_dual_cellulation = D
    D.meridian, D.longitude = (meridian, longitude)
    D.meridian_star, D.longitude_star = (mstar, lstar)

    def slope(onecycle):
        return vector([D.meridian_star(onecycle), D.longitude_star(onecycle)])
    D.slope = slope
    return (N, C, D, (mstar, lstar))