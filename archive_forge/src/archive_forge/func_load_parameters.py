import threading
import copy
import warnings
import re
import json
from collections import OrderedDict, defaultdict
import numpy as np
from ..base import mx_real_t, MXNetError
from .. import symbol, ndarray, initializer, np_symbol
from ..symbol import Symbol, load_json
from ..ndarray import NDArray
from .. import name as _name
from .parameter import Parameter, ParameterDict, DeferredInitializationError
from .utils import _indent, _brief_print_list, HookHandle
from .utils import _check_same_symbol_type, _check_all_np_ndarrays
from .. import numpy_extension as _mx_npx
from .. import numpy as _mx_np
from .. util import is_np_array, np_shape, np_array
def load_parameters(self, filename, ctx=None, allow_missing=False, ignore_extra=False, cast_dtype=False, dtype_source='current'):
    """Load parameters from file previously saved by `save_parameters`.

        Parameters
        ----------
        filename : str
            Path to parameter file.
        ctx : Context or list of Context, default cpu()
            Context(s) to initialize loaded parameters on.
        allow_missing : bool, default False
            Whether to silently skip loading parameters not represents in the file.
        ignore_extra : bool, default False
            Whether to silently ignore parameters from the file that are not
            present in this Block.
        cast_dtype : bool, default False
            Cast the data type of the NDArray loaded from the checkpoint to the dtype
            provided by the Parameter if any.
        dtype_source : str, default 'current'
            must be in {'current', 'saved'}
            Only valid if cast_dtype=True, specify the source of the dtype for casting
            the parameters
        References
        ----------
        `Saving and Loading Gluon Models         <https://mxnet.apache.org/api/python/docs/tutorials/packages/gluon/blocks/save_load_params.html>`_
        """
    if is_np_array():
        try:
            loaded = _mx_npx.load(filename)
        except MXNetError as e:
            err_msg = str(e)
            if 'is_np_shape' in err_msg:
                with np_array(False):
                    with np_shape(False):
                        loaded_nds = ndarray.load(filename)
                assert isinstance(loaded_nds, dict), 'expecting a dict type, got {}'.format(str(type(loaded_nds)))
                loaded = {k: loaded_nds[k].as_np_ndarray() for k in loaded_nds}
            else:
                raise ValueError(err_msg)
    else:
        loaded = ndarray.load(filename)
    params = self._collect_params_with_prefix()
    if not loaded and (not params):
        return
    if not any(('.' in i for i in loaded.keys())):
        loaded = None
        self.collect_params().load(filename, ctx, allow_missing, ignore_extra, self.prefix, cast_dtype=cast_dtype, dtype_source=dtype_source)
        return
    if not allow_missing:
        params_inv = defaultdict(list)
        for k, v in params.items():
            params_inv[v].append(k)
        for name, param in params.items():
            assert any((p in loaded for p in params_inv[param])), "Parameter '%s' is missing in file '%s', which contains parameters: %s. Set allow_missing=True to ignore missing parameters." % (name, filename, _brief_print_list(loaded.keys()))
    for name in loaded:
        if not ignore_extra and name not in params:
            raise ValueError("Parameter '%s' loaded from file '%s' is not present in ParameterDict, which contains parameters %s. Set ignore_extra=True to ignore. " % (name, filename, _brief_print_list(self._params.keys())))
        if name in params:
            params[name]._load_init(loaded[name], ctx, cast_dtype=cast_dtype, dtype_source=dtype_source)