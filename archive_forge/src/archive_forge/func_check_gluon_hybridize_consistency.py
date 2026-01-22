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
@use_np
def check_gluon_hybridize_consistency(net_builder, data_l, numpy_func=None, test_grad=True, rtol=0.0001, atol=0.0001):
    """Check whether a HybridBlock has consistent output between the hybridized
     v.s. non-hybridized versions

    The network should not contain any random number generators.

    Parameters
    ----------
    net_builder : function
        The builder of the HybridBlock that we are going to check the consistency.
        Inside the implementation, we will call net_builder() to construct the hybrid block.
        Also, the net_builder will need to support specifying the params
    data_l : list of mx.np.ndarray
        List of input ndarrays.
    numpy_func : function, optional
        The ground truth numpy function that has the same functionality as net_builder().
        Default None.
    test_grad : bool, optional
        Whether to test the consistency of the gradient. Default True.
    rtol : float, optional
        The relative error tolerance, default 1E-4. Default 1E-4.
    atol : float, optional
        The absolute error tolerance, default 1E-4. Default 1E-4.
    """

    class _NumpyParamDictInit(mx.init.Initializer):
        """Initializes parameters with the cached numpy ndarrays dictionary
        """

        def __init__(self, np_params):
            super(_NumpyParamDictInit, self).__init__()
            self._np_params = np_params

        def _init_weight(self, name, arr):
            arr[()] = self._np_params[name]
    saved_out_np = None
    saved_grad_np_l = None
    params_init = None
    use_autograd_flags = [False, True] if test_grad else [False]
    for hybridize in [False, True]:
        for use_autograd in use_autograd_flags:
            net = net_builder(prefix='net_')
            if params_init is None:
                net.initialize()
            else:
                net.initialize(params_init)
            if hybridize:
                net.hybridize()
            in_data_l = [ele.copy() for ele in data_l]
            if use_autograd:
                for ele in in_data_l:
                    ele.attach_grad()
                with mx.autograd.record():
                    out = net(*in_data_l)
                out.backward(out)
            else:
                out = net(*in_data_l)
            if params_init is None:
                np_params = {k: v.data().asnumpy() for k, v in net.collect_params().items()}
                params_init = _NumpyParamDictInit(np_params)
            if saved_out_np is None:
                saved_out_np = out.asnumpy()
            else:
                assert_almost_equal(out.asnumpy(), saved_out_np, rtol=rtol, atol=atol)
            if use_autograd:
                if saved_grad_np_l is None:
                    saved_grad_np_l = [ele.grad.asnumpy() for ele in in_data_l]
                else:
                    for data, saved_grad_np in zip(in_data_l, saved_grad_np_l):
                        assert_almost_equal(data.grad.asnumpy(), saved_grad_np, rtol=rtol, atol=atol)
    if numpy_func is not None:
        numpy_out = numpy_func(*[ele.asnumpy() for ele in data_l])
        assert_almost_equal(saved_out_np, numpy_out, rtol=rtol, atol=atol)