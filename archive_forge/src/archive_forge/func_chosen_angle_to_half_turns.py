from typing import Optional, overload
import numpy as np
import sympy
from cirq.value import type_alias
def chosen_angle_to_half_turns(half_turns: Optional[type_alias.TParamVal]=None, rads: Optional[float]=None, degs: Optional[float]=None, default: float=1.0) -> type_alias.TParamVal:
    """Returns a half_turns value based on the given arguments.

    At most one of half_turns, rads, degs must be specified. If none are
    specified, the output defaults to half_turns=1.

    Args:
        half_turns: The number of half turns to rotate by.
        rads: The number of radians to rotate by.
        degs: The number of degrees to rotate by
        default: The half turns angle to use if nothing else is specified.

    Returns:
        A number of half turns.

    Raises:
        ValueError: If more than one of `half_turn`, `rads`, or `degs` is given.
    """
    if len([1 for e in [half_turns, rads, degs] if e is not None]) > 1:
        raise ValueError('Redundant angle specification. Use ONE of half_turns, rads, or degs.')
    if rads is not None:
        return rads / np.pi
    if degs is not None:
        return degs / 180
    if half_turns is not None:
        return half_turns
    return default