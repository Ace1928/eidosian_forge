import contextlib
import functools
from llvmlite.ir import instructions, types, values
def shuffle_vector(self, vector1, vector2, mask, name=''):
    """
        Constructs a permutation of elements from *vector1* and *vector2*.
        Returns a new vector in the same length of *mask*.

        * *vector1* and *vector2* must have the same element type.
        * *mask* must be a constant vector of integer types.
        """
    instr = instructions.ShuffleVector(self.block, vector1, vector2, mask, name=name)
    self._insert(instr)
    return instr