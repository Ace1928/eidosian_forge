import pickle
import ctypes
import os
from ..ndarray import NDArray
from ..ndarray import _ndarray_cls
from ..base import _LIB, c_str
from ..base import check_call, mx_uint, py_str
from ..base import NDArrayHandle, KVStoreHandle
from .. import optimizer as opt
from .base import _ctype_key_value, _ctype_dict, KVStoreBase
def row_sparse_pull(self, key, out=None, priority=0, row_ids=None):
    """ Pulls a single RowSparseNDArray value or a sequence of RowSparseNDArray values         from the store with specified row_ids. When there is only one row_id, KVStoreRowSparsePull         is invoked just once and the result is broadcast to all the rest of outputs.

        `row_sparse_pull` is executed asynchronously after all previous
        `pull`/`row_sparse_pull` calls and the last `push` call for the
        same input key(s) are finished.

        The returned values are guaranteed to be the latest values in the store.

        Parameters
        ----------
        key : str, int, or sequence of str or int
            Keys.

        out: RowSparseNDArray or list of RowSparseNDArray or list of list of RowSparseNDArray
            Values corresponding to the keys. The stype is expected to be row_sparse

        priority : int, optional
            The priority of the pull operation.
            Higher priority pull operations are likely to be executed before
            other pull actions.

        row_ids : NDArray or list of NDArray
            The row_ids for which to pull for each value. Each row_id is an 1-D NDArray             whose values don't have to be unique nor sorted.

        Examples
        --------
        >>> shape = (3, 3)
        >>> kv.init('3', mx.nd.ones(shape).tostype('row_sparse'))
        >>> a = mx.nd.sparse.zeros('row_sparse', shape)
        >>> row_ids = mx.nd.array([0, 2], dtype='int64')
        >>> kv.row_sparse_pull('3', out=a, row_ids=row_ids)
        >>> print a.asnumpy()
        [[ 1.  1.  1.]
        [ 0.  0.  0.]
        [ 1.  1.  1.]]
        >>> duplicate_row_ids = mx.nd.array([2, 2], dtype='int64')
        >>> kv.row_sparse_pull('3', out=a, row_ids=duplicate_row_ids)
        >>> print a.asnumpy()
        [[ 0.  0.  0.]
        [ 0.  0.  0.]
        [ 1.  1.  1.]]
        >>> unsorted_row_ids = mx.nd.array([1, 0], dtype='int64')
        >>> kv.row_sparse_pull('3', out=a, row_ids=unsorted_row_ids)
        >>> print a.asnumpy()
        [[ 1.  1.  1.]
        [ 1.  1.  1.]
        [ 0.  0.  0.]]
        """
    assert out is not None
    assert row_ids is not None
    if isinstance(row_ids, NDArray):
        row_ids = [row_ids]
    assert isinstance(row_ids, list), 'row_ids should be NDArray or list of NDArray'
    first_out = out
    single_rowid = False
    if len(row_ids) == 1 and isinstance(out, list):
        single_rowid = True
        first_out = [out[0]]
    ckeys, cvals, use_str_keys = _ctype_key_value(key, first_out)
    _, crow_ids, _ = _ctype_key_value(key, row_ids)
    assert len(crow_ids) == len(cvals), "the number of row_ids doesn't match the number of values"
    if use_str_keys:
        check_call(_LIB.MXKVStorePullRowSparseEx(self.handle, mx_uint(len(ckeys)), ckeys, cvals, crow_ids, ctypes.c_int(priority)))
    else:
        check_call(_LIB.MXKVStorePullRowSparse(self.handle, mx_uint(len(ckeys)), ckeys, cvals, crow_ids, ctypes.c_int(priority)))
    if single_rowid:
        for out_i in out[1:]:
            out[0].copyto(out_i)