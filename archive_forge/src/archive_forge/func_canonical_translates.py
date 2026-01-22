from ...sage_helper import _within_sage
from ..upper_halfspace.finite_point import *
def canonical_translates(self, finitePoint):
    """
        TESTS::

            sage: from sage.all import *
            sage: t0 = CIF(RIF(2.3, 2.30000000001), 3.4)
            sage: t1 = CIF(4.32, RIF(5.43, 5.4300000001))
            sage: c = CuspTranslateEngine(t0, t1)
            sage: z = CIF(0.23, 0.43)
            sage: t = RIF(5)
            sage: for i in range(-2, 3): # doctest: +NUMERIC6
            ...     for j in range(-2, 3):
            ...         print(list(c.canonical_translates(FinitePoint(z + i * t0 + j * t1, t))))
            [FinitePoint(0.2300000000? + 0.430000000?*I, 5)]
            [FinitePoint(0.2300000000? + 0.430000000?*I, 5)]
            [FinitePoint(0.2300000000? + 0.430000000000000?*I, 5)]
            [FinitePoint(0.2300000000? + 0.430000000?*I, 5)]
            [FinitePoint(0.2300000000? + 0.430000000?*I, 5)]
            [FinitePoint(0.2300000000? + 0.430000000?*I, 5)]
            [FinitePoint(0.2300000000? + 0.430000000?*I, 5)]
            [FinitePoint(0.2300000000? + 0.430000000000000?*I, 5)]
            [FinitePoint(0.2300000000? + 0.430000000?*I, 5)]
            [FinitePoint(0.2300000000? + 0.430000000?*I, 5)]
            [FinitePoint(0.23000000000000? + 0.430000000?*I, 5)]
            [FinitePoint(0.230000000000000? + 0.430000000?*I, 5)]
            [FinitePoint(0.23000000000000001? + 0.43000000000000000?*I, 5)]
            [FinitePoint(0.230000000000000? + 0.430000000?*I, 5)]
            [FinitePoint(0.23000000000000? + 0.430000000?*I, 5)]
            [FinitePoint(0.2300000000? + 0.430000000?*I, 5)]
            [FinitePoint(0.2300000000? + 0.430000000?*I, 5)]
            [FinitePoint(0.2300000000? + 0.430000000000000?*I, 5)]
            [FinitePoint(0.2300000000? + 0.430000000?*I, 5)]
            [FinitePoint(0.2300000000? + 0.430000000?*I, 5)]
            [FinitePoint(0.2300000000? + 0.430000000?*I, 5)]
            [FinitePoint(0.2300000000? + 0.430000000?*I, 5)]
            [FinitePoint(0.2300000000? + 0.430000000000000?*I, 5)]
            [FinitePoint(0.2300000000? + 0.430000000?*I, 5)]
            [FinitePoint(0.2300000000? + 0.430000000?*I, 5)]

            sage: list(c.canonical_translates(FinitePoint(t0 / 2, t)))
            [FinitePoint(1.15000000000? + 1.7000000000000000?*I, 5), FinitePoint(-1.15000000000? - 1.7000000000000000?*I, 5)]

            sage: list(c.canonical_translates(FinitePoint(t1 / 2, t)))
            [FinitePoint(2.1600000000000002? + 2.7150000000?*I, 5), FinitePoint(-2.1600000000000002? - 2.7150000000?*I, 5)]

            sage: list(c.canonical_translates(FinitePoint(t0 / 2 + t1 / 2, t)))
            [FinitePoint(3.31000000001? + 4.4150000000?*I, 5), FinitePoint(-1.01000000000? - 1.015000000?*I, 5), FinitePoint(1.01000000000? + 1.0150000000?*I, 5), FinitePoint(-3.3100000000? - 4.415000000?*I, 5)]
        """
    for z in self._canonical_translates(finitePoint.z):
        yield FinitePoint(z, finitePoint.t)