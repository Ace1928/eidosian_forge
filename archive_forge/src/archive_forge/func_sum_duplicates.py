import numpy
import cupy
from cupy import _core
from cupyx import cusparse
from cupyx.scipy.sparse import _base
from cupyx.scipy.sparse import _csc
from cupyx.scipy.sparse import _csr
from cupyx.scipy.sparse import _data as sparse_data
from cupyx.scipy.sparse import _util
from cupyx.scipy.sparse import _sputils
def sum_duplicates(self):
    """Eliminate duplicate matrix entries by adding them together.

        .. warning::
            When sorting the indices, CuPy follows the convention of cuSPARSE,
            which is different from that of SciPy. Therefore, the order of the
            output indices may differ:

            .. code-block:: python

                >>> #     1 0 0
                >>> # A = 1 1 0
                >>> #     1 1 1
                >>> data = cupy.array([1, 1, 1, 1, 1, 1], 'f')
                >>> row = cupy.array([0, 1, 1, 2, 2, 2], 'i')
                >>> col = cupy.array([0, 0, 1, 0, 1, 2], 'i')
                >>> A = cupyx.scipy.sparse.coo_matrix((data, (row, col)),
                ...                                   shape=(3, 3))
                >>> a = A.get()
                >>> A.sum_duplicates()
                >>> a.sum_duplicates()  # a is scipy.sparse.coo_matrix
                >>> A.row
                array([0, 1, 1, 2, 2, 2], dtype=int32)
                >>> a.row
                array([0, 1, 2, 1, 2, 2], dtype=int32)
                >>> A.col
                array([0, 0, 1, 0, 1, 2], dtype=int32)
                >>> a.col
                array([0, 0, 0, 1, 1, 2], dtype=int32)

        .. warning::
            Calling this function might synchronize the device.

        .. seealso::
           :meth:`scipy.sparse.coo_matrix.sum_duplicates`

        """
    if self.has_canonical_format:
        return
    keys = cupy.stack([self.col, self.row])
    order = cupy.lexsort(keys)
    src_data = self.data[order]
    src_row = self.row[order]
    src_col = self.col[order]
    diff = self._sum_duplicates_diff(src_row, src_col, size=self.row.size)
    if diff[1:].all():
        data = src_data
        row = src_row
        col = src_col
    else:
        index = cupy.cumsum(diff, dtype='i')
        size = int(index[-1]) + 1
        data = cupy.zeros(size, dtype=self.data.dtype)
        row = cupy.empty(size, dtype='i')
        col = cupy.empty(size, dtype='i')
        if self.data.dtype.kind == 'f':
            cupy.ElementwiseKernel('T src_data, int32 src_row, int32 src_col, int32 index', 'raw T data, raw int32 row, raw int32 col', '\n                    atomicAdd(&data[index], src_data);\n                    row[index] = src_row;\n                    col[index] = src_col;\n                    ', 'cupyx_scipy_sparse_coo_sum_duplicates_assign')(src_data, src_row, src_col, index, data, row, col)
        elif self.data.dtype.kind == 'c':
            cupy.ElementwiseKernel('T src_real, T src_imag, int32 src_row, int32 src_col, int32 index', 'raw T real, raw T imag, raw int32 row, raw int32 col', '\n                    atomicAdd(&real[index], src_real);\n                    atomicAdd(&imag[index], src_imag);\n                    row[index] = src_row;\n                    col[index] = src_col;\n                    ', 'cupyx_scipy_sparse_coo_sum_duplicates_assign_complex')(src_data.real, src_data.imag, src_row, src_col, index, data.real, data.imag, row, col)
    self.data = data
    self.row = row
    self.col = col
    self.has_canonical_format = True