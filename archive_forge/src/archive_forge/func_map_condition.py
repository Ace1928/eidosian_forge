from __future__ import annotations
import typing
from .bit import Bit
from .classical import expr
from .classicalregister import ClassicalRegister, Clbit
def map_condition(self, condition, /, *, allow_reorder=False):
    """Map the given ``condition`` so that it only references variables in the destination
        circuit (as given to this class on initialisation).

        If ``allow_reorder`` is ``True``, then when a legacy condition (the two-tuple form) is made
        on a register that has a counterpart in the destination with all the same (mapped) bits but
        in a different order, then that register will be used and the value suitably modified to
        make the equality condition work.  This is maintaining legacy (tested) behaviour of
        :meth:`.DAGCircuit.compose`; nowhere else does this, and in general this would require *far*
        more complex classical rewriting than Terra needs to worry about in the full expression era.
        """
    if condition is None:
        return None
    if isinstance(condition, expr.Expr):
        return self.map_expr(condition)
    target, value = condition
    if isinstance(target, Clbit):
        return (self.bit_map[target], value)
    if not allow_reorder:
        return (self._map_register(target), value)
    mapped_bits_order = [self.bit_map[bit] for bit in target]
    mapped_bits_set = set(mapped_bits_order)
    for register in self.target_cregs:
        if mapped_bits_set == set(register):
            mapped_theirs = register
            break
    else:
        if self.add_register is None:
            raise self.exc_type(f"Register '{target.name}' has no counterpart in the destination.")
        mapped_theirs = ClassicalRegister(bits=mapped_bits_order)
        self.add_register(mapped_theirs)
    new_order = {bit: i for i, bit in enumerate(mapped_bits_order)}
    value_bits = f'{value:0{len(target)}b}'[::-1]
    mapped_value = int(''.join((value_bits[new_order[bit]] for bit in mapped_theirs))[::-1], 2)
    return (mapped_theirs, mapped_value)