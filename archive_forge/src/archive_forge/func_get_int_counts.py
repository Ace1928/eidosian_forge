from __future__ import annotations
from collections import defaultdict
from functools import partial
from itertools import chain, repeat
from typing import Callable, Iterable, Literal, Mapping
import numpy as np
from numpy.typing import NDArray
from qiskit.result import Counts
from .shape import ShapedMixin, ShapeInput, shape_tuple
def get_int_counts(self, loc: int | tuple[int, ...] | None=None) -> dict[int, int]:
    """Return a counts dictionary, where bitstrings are stored as ``int``\\s.

        Args:
            loc: Which entry of this array to return a dictionary for. If ``None``, counts from
                all positions in this array are unioned together.

        Returns:
            A dictionary mapping ``ints`` to the number of occurrences of that ``int``.

        """
    converter = partial(self._bytes_to_int, mask=2 ** self.num_bits - 1)
    return self._get_counts(loc=loc, converter=converter)