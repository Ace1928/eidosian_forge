import math
from itertools import chain
from operator import add, mul, truediv, sub, pow
from .pyutil import defaultkeydict, deprecated
from ._expr_deprecated import _mk_PiecewisePoly, _mk_Poly  # noqa
def dedimensionalisation(self, unit_registry, variables={}, backend=math):
    """Create an instance with consistent units from a unit_registry

        Parameters
        ----------
        unit_registry : dict
        variables : dict
        backend : module

        Examples
        --------
        >>> class Pressure(Expr):
        ...     argument_names = ('n',)
        ...     parameter_keys = ('temperature', 'volume', 'R')
        ...     def __call__(self, variables, backend=math, **kwargs):
        ...         n, = self.all_args(variables, backend=backend)
        ...         T, V, R = self.all_params(variables, backend=backend)
        ...         return n*R*T/V
        ...
        >>> from chempy.units import SI_base_registry, default_units as u
        >>> p = Pressure([2*u.micromole])
        >>> units, d = p.dedimensionalisation(SI_base_registry)
        >>> units[0] == 1e6*u.micromole
        True
        >>> d.args[0] == 2e-6
        True


        Returns
        -------
        new_units: list of units of the dedimensionalised args.
        self.__class__ instance: with dedimensioanlised arguments

        """
    from ..units import default_unit_in_registry, to_unitless, unitless_in_registry
    new_units = []
    if self.args is None:
        unitless_args = None
    else:
        unitless_args = []
        units = [None if isinstance(arg, Expr) else default_unit_in_registry(arg, unit_registry) for arg in self.all_args(variables, backend=backend, evaluate=False)]
        for arg, unit in zip(self.all_args(variables, backend=backend, evaluate=False), units):
            if isinstance(arg, Expr):
                if unit is not None:
                    raise ValueError()
                _unit, _dedim = arg.dedimensionalisation(unit_registry, variables, backend=backend)
            else:
                _unit, _dedim = (unit, to_unitless(arg, unit))
            new_units.append(_unit)
            unitless_args.append(_dedim)
    instance = self.__class__(unitless_args, self.unique_keys)
    if self.argument_defaults is not None:
        instance.argument_defaults = tuple((unitless_in_registry(arg, unit_registry) for arg in self.argument_defaults))
    return (new_units, instance)