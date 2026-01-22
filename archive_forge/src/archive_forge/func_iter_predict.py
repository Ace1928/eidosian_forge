import time
import logging
import warnings
import numpy as np
from .. import metric
from .. import ndarray
from ..context import cpu
from ..model import BatchEndParam
from ..initializer import Uniform
from ..io import DataDesc, DataIter, DataBatch
from ..base import _as_list
def iter_predict(self, eval_data, num_batch=None, reset=True, sparse_row_id_fn=None):
    """Iterates over predictions.

        Examples
        --------
        >>> for pred, i_batch, batch in module.iter_predict(eval_data):
        ...     # pred is a list of outputs from the module
        ...     # i_batch is a integer
        ...     # batch is the data batch from the data iterator

        Parameters
        ----------
        eval_data : DataIter
            Evaluation data to run prediction on.
        num_batch : int
            Default is ``None``, indicating running all the batches in the data iterator.
        reset : bool
            Default is ``True``, indicating whether we should reset the data iter before start
            doing prediction.
        sparse_row_id_fn : A callback function
            The function  takes `data_batch` as an input and returns a dict of
            str -> NDArray. The resulting dict is used for pulling row_sparse
            parameters from the kvstore, where the str key is the name of the param,
            and the value is the row id of the param to pull.
        """
    assert self.binded and self.params_initialized
    if reset:
        eval_data.reset()
    for nbatch, eval_batch in enumerate(eval_data):
        if num_batch is not None and nbatch == num_batch:
            break
        self.prepare(eval_batch, sparse_row_id_fn=sparse_row_id_fn)
        self.forward(eval_batch, is_train=False)
        pad = eval_batch.pad
        outputs = [out[0:out.shape[0] - pad] for out in self.get_outputs()]
        yield (outputs, nbatch, eval_batch)