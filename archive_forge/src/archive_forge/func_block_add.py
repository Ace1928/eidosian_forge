import numpy as np
import scipy.sparse as sp
from cvxpy.interface.numpy_interface.ndarray_interface import NDArrayInterface
def block_add(self, matrix, block, vert_offset, horiz_offset, rows, cols, vert_step: int=1, horiz_step: int=1) -> None:
    """Add the block to a slice of the matrix.

        Args:
            matrix: The matrix the block will be added to.
            block: The matrix/scalar to be added.
            vert_offset: The starting row for the matrix slice.
            horiz_offset: The starting column for the matrix slice.
            rows: The height of the block.
            cols: The width of the block.
            vert_step: The row step size for the matrix slice.
            horiz_step: The column step size for the matrix slice.
        """
    block = self._format_block(matrix, block, rows, cols)
    slice_ = [slice(vert_offset, rows + vert_offset, vert_step), slice(horiz_offset, horiz_offset + cols, horiz_step)]
    matrix[slice_[0], slice_[1]] = matrix[slice_[0], slice_[1]] + block