from pyomo.common.dependencies import mpi4py
from pyomo.contrib.pynumero.sparse import BlockVector
from .base_block import BaseBlockVector
from .block_vector import NotFullyDefinedBlockVectorError
from .block_vector import assert_block_structure as block_vector_assert_block_structure
import numpy as np
import operator
def finalize_block_sizes(self, broadcast=True, block_sizes=None):
    """
        Only set broadcast=False if you know what you are doing!

        Parameters
        ----------
        broadcast: bool
        block_sizes: None or np.ndarray
        """
    if broadcast:
        self.broadcast_block_sizes()
    else:
        self._undefined_brows = set()
        self._brow_lengths = block_sizes
        self._broadcasted = True