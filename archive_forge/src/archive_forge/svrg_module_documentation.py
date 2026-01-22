import time
import logging
import mxnet as mx
from mxnet.module import Module
from .svrg_optimizer import _SVRGOptimizer
Prepares two modules for processing a data batch.

        Usually involves switching bucket and reshaping.
        For modules that contain `row_sparse` parameters in KVStore,
        it prepares the `row_sparse` parameters based on the sparse_row_id_fn.

        When KVStore is used to update parameters for multi-device or multi-machine training,
        a copy of the parameters are stored in KVStore. Note that for `row_sparse` parameters,
        the `update()` updates the copy of parameters in KVStore, but doesn't broadcast
        the updated parameters to all devices / machines. The `prepare` function is used to
        broadcast `row_sparse` parameters with the next batch of data.

        Parameters
        ----------
        data_batch : DataBatch
            The current batch of data for forward computation.

        sparse_row_id_fn : A callback function
            The function  takes `data_batch` as an input and returns a dict of
            str -> NDArray. The resulting dict is used for pulling row_sparse
            parameters from the kvstore, where the str key is the name of the param,
            and the value is the row id of the param to pull.
        