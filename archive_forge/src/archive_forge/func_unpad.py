import itertools
import math
from typing import (
import numpy
from ..types import (
from ..util import get_array_module, is_xp_array, to_numpy
from .cblas import CBlas
def unpad(self, padded: Array3d, lengths: List[int]) -> List2d:
    """The reverse/backward operation of the `pad` function: transform an
        array back into a list of arrays, each with their original length.
        """
    output = []
    for i, length in enumerate(lengths):
        output.append(padded[i, :length])
    return cast(List2d, output)