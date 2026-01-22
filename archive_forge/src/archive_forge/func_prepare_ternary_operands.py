from deprecated import deprecated
from deprecated.sphinx import versionadded
from numbers import Real
from typing import Callable, Mapping, Optional, Tuple, Union, Iterable, no_type_check
import numpy as np
from pyquil.quilatom import (
from pyquil.quilbase import (
def prepare_ternary_operands(classical_reg1: MemoryReferenceDesignator, classical_reg2: MemoryReferenceDesignator, classical_reg3: Union[MemoryReferenceDesignator, int, float]) -> Tuple[MemoryReference, MemoryReference, Union[MemoryReference, int, float]]:
    """
    Helper function for typechecking / type-coercing arguments to constructors for ternary
    classical operators.

    :param classical_reg1: Specifier for the classical memory address to be modified.
    :param classical_reg2: Specifier for the left operand: a classical memory address.
    :param classical_reg3: Specifier for the right operand: a classical memory address or an
        immediate value.
    :return: A triple of pyQuil objects suitable for use as operands.
    """
    if isinstance(classical_reg1, int):
        raise TypeError('Target operand of comparison must be a memory address')
    classical_reg1 = unpack_classical_reg(classical_reg1)
    if isinstance(classical_reg2, int):
        raise TypeError('Left operand of comparison must be a memory address')
    classical_reg2 = unpack_classical_reg(classical_reg2)
    if not isinstance(classical_reg3, (float, int)):
        classical_reg3 = unpack_classical_reg(classical_reg3)
    return (classical_reg1, classical_reg2, classical_reg3)