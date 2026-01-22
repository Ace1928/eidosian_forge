from .polished_reps import ManifoldGroup
from .fundamental_polyhedron import *

    Returns the representation

        rho: pi_1(manifold) -> (P)SL(2, ComplexIntervalField)

    determined by the given shape_intervals.  If shape_intervals
    contains an exact solution z0 to the gluing equations with
    corresponding holonomy representation rho0, then for all g the
    ComplexIntervalField matrix rho(g) contains rho0(g)::

        sage: M = Manifold('m004(1,2)')
        sage: success, shapes = M.verify_hyperbolicity(bits_prec=53)
        sage: success
        True
        sage: rho = holonomy_from_shape_intervals(M, shapes)
        sage: (rho('a').det() - 1).contains_zero()
        True

    Of course, for long words the matrix entries will smear out::

        sage: diameter(rho('a')).log10() # doctest: +NUMERIC0
        -10.9576580520835
        sage: diameter(rho(10*'abAB')).log10() # doctest: +NUMERIC0
        -8.39987365046327
    