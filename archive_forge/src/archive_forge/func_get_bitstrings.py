from __future__ import annotations
from collections import defaultdict
from functools import partial
from itertools import chain, repeat
from typing import Callable, Iterable, Literal, Mapping
import numpy as np
from numpy.typing import NDArray
from qiskit.result import Counts
from .shape import ShapedMixin, ShapeInput, shape_tuple
def get_bitstrings(self, loc: int | tuple[int, ...] | None=None) -> list[str]:
    """Return a list of bitstrings.

        Args:
            loc: Which entry of this array to return a dictionary for. If ``None``, counts from
                all positions in this array are unioned together.

        Returns:
            A list of bitstrings.
        """
    mask = 2 ** self.num_bits - 1
    converter = partial(self._bytes_to_bitstring, num_bits=self.num_bits, mask=mask)
    arr = self._array.reshape(-1, self._array.shape[-1]) if loc is None else self._array[loc]
    return [converter(shot_row.tobytes()) for shot_row in arr]