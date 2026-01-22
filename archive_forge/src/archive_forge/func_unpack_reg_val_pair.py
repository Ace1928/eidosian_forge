from deprecated import deprecated
from deprecated.sphinx import versionadded
from numbers import Real
from typing import Callable, Mapping, Optional, Tuple, Union, Iterable, no_type_check
import numpy as np
from pyquil.quilatom import (
from pyquil.quilbase import (
def unpack_reg_val_pair(classical_reg1: MemoryReferenceDesignator, classical_reg2: Union[MemoryReferenceDesignator, int, float]) -> Tuple[MemoryReference, Union[MemoryReference, int, float]]:
    """
    Helper function for typechecking / type-coercing arguments to constructors for binary classical
    operators.

    :param classical_reg1: Specifier for the classical memory address to be modified.
    :param classical_reg2: Specifier for the second argument: a classical memory address or an
        immediate value.
    :return: A pair of pyQuil objects suitable for use as operands.
    """
    left = unpack_classical_reg(classical_reg1)
    if isinstance(classical_reg2, (float, int)):
        return (left, classical_reg2)
    return (left, unpack_classical_reg(classical_reg2))