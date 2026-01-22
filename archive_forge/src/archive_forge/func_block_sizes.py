import operator
from ..dependencies import numpy as np
from .base_block import BaseBlockVector
def block_sizes(self, copy=True):
    """
        Returns 1D-Array with sizes of individual blocks in this BlockVector
        """
    assert_block_structure(self)
    if copy:
        return self._brow_lengths.copy()
    return self._brow_lengths