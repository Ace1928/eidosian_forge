from functools import reduce
from collections.abc import Iterable
from typing import Optional
from sympy import default_sort_key
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.sorting import ordered
from sympy.core.sympify import sympify
from sympy.matrices.common import NonInvertibleMatrixError
from sympy.physics.units.dimensions import Dimension, DimensionSystem
from sympy.physics.units.prefixes import Prefix
from sympy.physics.units.quantities import Quantity
from sympy.physics.units.unitsystem import UnitSystem
from sympy.utilities.iterables import sift
def quantity_simplify(expr, across_dimensions: bool=False, unit_system=None):
    """Return an equivalent expression in which prefixes are replaced
    with numerical values and all units of a given dimension are the
    unified in a canonical manner by default. `across_dimensions` allows
    for units of different dimensions to be simplified together.

    `unit_system` must be specified if `across_dimensions` is True.

    Examples
    ========

    >>> from sympy.physics.units.util import quantity_simplify
    >>> from sympy.physics.units.prefixes import kilo
    >>> from sympy.physics.units import foot, inch, joule, coulomb
    >>> quantity_simplify(kilo*foot*inch)
    250*foot**2/3
    >>> quantity_simplify(foot - 6*inch)
    foot/2
    >>> quantity_simplify(5*joule/coulomb, across_dimensions=True, unit_system="SI")
    5*volt
    """
    if expr.is_Atom or not expr.has(Prefix, Quantity):
        return expr
    p = expr.atoms(Prefix)
    expr = expr.xreplace({p: p.scale_factor for p in p})
    d = sift(expr.atoms(Quantity), lambda i: i.dimension)
    for k in d:
        if len(d[k]) == 1:
            continue
        v = list(ordered(d[k]))
        ref = v[0] / v[0].scale_factor
        expr = expr.xreplace({vi: ref * vi.scale_factor for vi in v[1:]})
    if across_dimensions:
        if unit_system is None:
            raise ValueError('unit_system must be specified if across_dimensions is True')
        unit_system = UnitSystem.get_unit_system(unit_system)
        dimension_system: DimensionSystem = unit_system.get_dimension_system()
        dim_expr = unit_system.get_dimensional_expr(expr)
        dim_deps = dimension_system.get_dimensional_dependencies(dim_expr, mark_dimensionless=True)
        target_dimension: Optional[Dimension] = None
        for ds_dim, ds_dim_deps in dimension_system.dimensional_dependencies.items():
            if ds_dim_deps == dim_deps:
                target_dimension = ds_dim
                break
        if target_dimension is None:
            return expr
        target_unit = unit_system.derived_units.get(target_dimension)
        if target_unit:
            expr = convert_to(expr, target_unit, unit_system)
    return expr