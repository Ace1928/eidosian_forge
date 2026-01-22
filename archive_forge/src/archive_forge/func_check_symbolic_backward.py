import time
import gzip
import struct
import traceback
import numbers
import sys
import os
import platform
import errno
import logging
import bz2
import zipfile
import json
from contextlib import contextmanager
from collections import OrderedDict
import numpy as np
import numpy.testing as npt
import numpy.random as rnd
import mxnet as mx
from .context import Context, current_context
from .ndarray.ndarray import _STORAGE_TYPE_STR_TO_ID
from .ndarray import array
from .symbol import Symbol
from .symbol.numpy import _Symbol as np_symbol
from .util import use_np, getenv, setenv  # pylint: disable=unused-import
from .runtime import Features
from .numpy_extension import get_cuda_compute_capability
def check_symbolic_backward(sym, location, out_grads, expected, rtol=None, atol=None, aux_states=None, grad_req='write', ctx=None, grad_stypes=None, equal_nan=False, dtype=default_dtype()):
    """Compares a symbol's backward results with the expected ones.
    Prints error messages if the backward results are not the same as the expected results.

    Parameters
    ---------
    sym : Symbol
        output symbol
    location : list of np.ndarray or dict of str to np.ndarray
        The evaluation point

        - if type is list of np.ndarray
            Contains all the NumPy arrays corresponding to ``mx.sym.list_arguments``.
        - if type is dict of str to np.ndarray
            Contains the mapping between argument names and their values.
    out_grads : None or list of np.ndarray or dict of str to np.ndarray
        NumPys arrays corresponding to sym.outputs for incomming gradient.

        - if type is list of np.ndarray
            Contains arrays corresponding to ``exe.outputs``.
        - if type is dict of str to np.ndarray
            contains mapping between mxnet.sym.list_output() and Executor.outputs
    expected : list of np.ndarray or dict of str to np.ndarray
        expected gradient values

        - if type is list of np.ndarray
            Contains arrays corresponding to exe.grad_arrays
        - if type is dict of str to np.ndarray
            Contains mapping between ``sym.list_arguments()`` and exe.outputs.
    rtol : None or float
        The relative threshold. Default threshold will be used if set to ``None``.
    atol : None or float
        The absolute threshold. Default threshold will be used if set to ``None``.
    aux_states : list of np.ndarray or dict of str to np.ndarray
    grad_req : str or list of str or dict of str to str, optional
        Gradient requirements. 'write', 'add' or 'null'.
    ctx : Context, optional
        Running context.
    grad_stypes: dict of str->str
        dictionary of mapping argument name to stype for the gradient
    equal_nan: Boolean
        if True, `nan` is a valid value for checking equivalency (ie `nan` == `nan`)
    dtype: np.float16 or np.float32 or np.float64
        Datatype for mx.nd.array.

    Example
    -------
    >>> lhs = mx.symbol.Variable('lhs')
    >>> rhs = mx.symbol.Variable('rhs')
    >>> sym_add = mx.symbol.elemwise_add(lhs, rhs)
    >>> mat1 = np.array([[1, 2], [3, 4]])
    >>> mat2 = np.array([[5, 6], [7, 8]])
    >>> grad1 = mx.nd.zeros(shape)
    >>> grad2 = mx.nd.zeros(shape)
    >>> exec_add = sym_add.bind(default_context(), args={'lhs': mat1, 'rhs': mat2},
    ... args_grad={'lhs': grad1, 'rhs': grad2}, grad_req={'lhs': 'write', 'rhs': 'write'})
    >>> exec_add.forward(is_train=True)
    >>> ograd = mx.nd.ones(shape)
    >>> grad_expected = ograd.copy().asnumpy()
    >>> check_symbolic_backward(sym_add, [mat1, mat2], [ograd], [grad_expected, grad_expected])
    """
    assert dtype == 'asnumpy' or dtype in (np.float16, np.float32, np.float64)
    if ctx is None:
        ctx = default_context()
    location = _parse_location(sym=sym, location=location, ctx=ctx, dtype=dtype)
    aux_states = _parse_aux_states(sym=sym, aux_states=aux_states, ctx=ctx, dtype=dtype)
    if isinstance(expected, (list, tuple)):
        expected = {k: v for k, v in zip(sym.list_arguments(), expected)}
    args_grad_npy = {k: np.random.normal(size=v.shape) for k, v in _sorted_items(expected)}
    args_grad_data = {}
    for k, v in args_grad_npy.items():
        nd = mx.nd.array(v, ctx=ctx, dtype=expected[k].dtype if dtype == 'asnumpy' else dtype)
        if grad_stypes is not None and k in grad_stypes:
            stype = grad_stypes[k]
            if stype is not None and stype != 'default':
                out = create_sparse_array(v.shape, stype, density=0.0)
            else:
                out = nd
            args_grad_data[k] = out
        else:
            args_grad_data[k] = nd
    if isinstance(grad_req, str):
        grad_req = {k: grad_req for k in sym.list_arguments()}
    elif isinstance(grad_req, (list, tuple)):
        grad_req = {k: v for k, v in zip(sym.list_arguments(), grad_req)}
    executor = sym.bind(ctx=ctx, args=location, args_grad=args_grad_data, aux_states=aux_states, grad_req=grad_req)
    executor.forward(is_train=True)
    if isinstance(out_grads, (tuple, list)):
        outg = list()
        for arr in out_grads:
            if isinstance(arr, np.ndarray):
                outg.append(mx.nd.array(arr, ctx=ctx, dtype=arr.dtype if dtype == 'asnumpy' else dtype))
            else:
                outg.append(arr)
        out_grads = outg
    elif isinstance(out_grads, dict):
        outg = dict()
        for k, v in out_grads.items():
            if isinstance(v, np.ndarray):
                outg[k] = mx.nd.array(v, ctx=ctx, dtype=v.dtype if dtype == 'asnumpy' else dtype)
            else:
                outg[k] = v
        out_grads = outg
    else:
        assert out_grads is None
    executor.backward(out_grads)
    grads = args_grad_data
    for name in expected:
        if grad_req[name] == 'write':
            assert_almost_equal(expected[name], grads[name], rtol, atol, ('EXPECTED_%s' % name, 'BACKWARD_%s' % name), equal_nan=equal_nan)
        elif grad_req[name] == 'add':
            grad = grads[name].asnumpy() if isinstance(grads[name], mx.nd.NDArray) else grads[name]
            assert_almost_equal(expected[name], grad - args_grad_npy[name], rtol, atol, ('EXPECTED_%s' % name, 'BACKWARD_%s' % name), equal_nan=equal_nan)
        elif grad_req[name] == 'null':
            assert_almost_equal(args_grad_npy[name], grads[name], rtol, atol, ('EXPECTED_%s' % name, 'BACKWARD_%s' % name), equal_nan=equal_nan)
        else:
            raise ValueError('Invalid grad_req %s for argument %s' % (grad_req[name], name))
    return args_grad_data