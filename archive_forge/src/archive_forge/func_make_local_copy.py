from pyomo.common.dependencies import mpi4py
from pyomo.contrib.pynumero.sparse import BlockVector
from .base_block import BaseBlockVector
from .block_vector import NotFullyDefinedBlockVectorError
from .block_vector import assert_block_structure as block_vector_assert_block_structure
import numpy as np
import operator
def make_local_copy(self):
    """
        Copies the MPIBlockVector into a BlockVector

        Returns
        -------
        BlockVector
        """
    assert_block_structure(self)
    if not self.is_broadcasted():
        self.broadcast_block_sizes()
    result = self.make_local_structure_copy()
    local_data = np.zeros(self.size)
    global_data = np.zeros(self.size)
    offset = 0
    rank = self._mpiw.Get_rank()
    if rank == 0:
        block_indices = set(self._owned_blocks)
    else:
        block_indices = set(self._unique_owned_blocks)
    for ndx in range(self.nblocks):
        if ndx in block_indices:
            blk = self.get_block(ndx)
            if isinstance(blk, BlockVector):
                local_data[offset:offset + self.get_block_size(ndx)] = blk.flatten()
            elif isinstance(blk, np.ndarray):
                local_data[offset:offset + self.get_block_size(ndx)] = blk
            else:
                raise ValueError('Unrecognized block type')
        offset += self.get_block_size(ndx)
    self._mpiw.Allreduce(local_data, global_data)
    result.copyfrom(global_data)
    return result