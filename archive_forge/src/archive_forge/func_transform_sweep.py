from typing import Any, Callable, List, Optional, Tuple, Union, TYPE_CHECKING
import numbers
import sympy
from cirq import protocols
from cirq.study import resolver, sweeps, sweepable
def transform_sweep(self, sweep: Union[sweeps.Sweep, List[resolver.ParamResolver]]) -> sweeps.Sweep:
    """Returns a sweep to use with a circuit flattened earlier with
        `cirq.flatten`.

        If `sweep` sweeps symbol `a` over (1.0, 2.0, 3.0) and this
        `ExpressionMap` maps `a/2+1` to the symbol `'<a/2 + 1>'` then this
        method returns a sweep that sweeps symbol `'<a/2 + 1>'` over
        (1.5, 2, 2.5).

        See `cirq.flatten` for an example.

        Args:
            sweep: The sweep to transform.
        """
    sweep = sweepable.to_sweep(sweep)
    param_list: List[resolver.ParamDictType] = []
    for r in sweep:
        param_dict: resolver.ParamDictType = {}
        for formula, sym in self.items():
            if isinstance(sym, (sympy.Symbol, str)):
                param_dict[str(sym)] = protocols.resolve_parameters(formula, r)
        param_list.append(param_dict)
    return sweeps.ListSweep(param_list)