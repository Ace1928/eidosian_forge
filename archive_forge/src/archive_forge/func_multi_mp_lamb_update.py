import math
import numpy as np
import mxnet as mx
from ..context import current_context
from ..random import uniform
from ..base import _as_list
from . import ndarray
def multi_mp_lamb_update(weights, grads, mean, var, weights32, step_count, lrs, wds, out=None, num_tensors=0, **kwargs):
    """Given a list of gradients, update weights, mean and variance of multiple tensors
    following LAMB Optimizer implementation, and using Mixed-Precision.

    Parameters
    ----------
    weights : List of NDArrays containing the input weights of multiple tensors

    grads : List of NDArrays containing input gradients

    mean : List of NDArrays containing mean of multiple tensors to be updated

    var : List of NDArrays containing variance of multiple tensors to be updated

    weights32 : Master copy of weights in FP32

    step_count : List of scalars with the number of update step for each tensor

    lrs : List of learning rates (one for each tensor)

    wds : List of weight decays (one for each tensor)

    out: List of NDArrays where the updated weights will be stored

    num_tensors : Number of NDArrays/tensors in the list
    """
    if not num_tensors:
        num_tensors = len(weights)
    temp_list = _flatten_list(zip(weights, grads, mean, var, weights32))
    return ndarray._internal._multi_mp_lamb_update(*temp_list, out=out, num_tensors=num_tensors, step_count=step_count, learning_rates=lrs, wds=wds, **kwargs)