from array import array
import ctypes
import warnings
from numbers import Number
import numpy as _numpy  # pylint: disable=relative-import
from ..attribute import AttrScope
from ..base import _LIB, numeric_types, c_array, c_array_buf, c_str, c_str_array, c_handle_array
from ..base import mx_uint, py_str, string_types, integer_types, mx_int, mx_int64
from ..base import NDArrayHandle, ExecutorHandle, SymbolHandle
from ..base import check_call, MXNetError, NotImplementedForSymbol
from ..context import Context, current_context
from ..ndarray import NDArray, _DTYPE_NP_TO_MX, _DTYPE_MX_TO_NP, _GRAD_REQ_MAP
from ..ndarray.ndarray import _STORAGE_TYPE_STR_TO_ID, _int64_enabled, _SIGNED_INT32_UPPER_LIMIT
from ..ndarray import _ndarray_cls
from ..executor import Executor
from . import _internal
from . import op
from ._internal import SymbolBase, _set_symbol_class
from ..util import is_np_shape
def simple_bind(self, ctx, grad_req='write', type_dict=None, stype_dict=None, group2ctx=None, shared_arg_names=None, shared_exec=None, shared_buffer=None, **kwargs):
    """Bind current symbol to get an executor, allocate all the arguments needed.
        Allows specifying data types.

        This function simplifies the binding procedure. You need to specify only input data shapes.
        Before binding the executor, the function allocates arguments and auxiliary states
        that were not explicitly specified. Allows specifying data types.

        Example
        -------
        >>> x = mx.sym.Variable('x')
        >>> y = mx.sym.FullyConnected(x, num_hidden=4)
        >>> exe = y.simple_bind(mx.cpu(), x=(5,4), grad_req='null')
        >>> exe.forward()
        [<NDArray 5x4 @cpu(0)>]
        >>> exe.outputs[0].asnumpy()
        array([[ 0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.]], dtype=float32)
        >>> exe.arg_arrays
        [<NDArray 5x4 @cpu(0)>, <NDArray 4x4 @cpu(0)>, <NDArray 4 @cpu(0)>]
        >>> exe.grad_arrays
        [<NDArray 5x4 @cpu(0)>, <NDArray 4x4 @cpu(0)>, <NDArray 4 @cpu(0)>]

        Parameters
        ----------
        ctx : Context
            The device context the generated executor to run on.

        grad_req: string
            {'write', 'add', 'null'}, or list of str or dict of str to str, optional
            To specify how we should update the gradient to the `args_grad`.

            - 'write' means every time gradient is written to specified `args_grad` NDArray.
            - 'add' means every time gradient is added to the specified NDArray.
            - 'null' means no action is taken, the gradient may not be calculated.

        type_dict  : Dict of str->numpy.dtype
            Input type dictionary, name->dtype

        stype_dict  : Dict of str->str
            Input storage type dictionary, name->storage_type

        group2ctx : Dict of string to mx.Context
            The dict mapping the `ctx_group` attribute to the context assignment.

        shared_arg_names : List of string
            The argument names whose `NDArray` of shared_exec can be reused for initializing
            the current executor.

        shared_exec : Executor
            The executor whose arg_arrays, arg_arrays, grad_arrays, and aux_arrays can be
            reused for initializing the current executor.

        shared_buffer : Dict of string to `NDArray`
            The dict mapping argument names to the `NDArray` that can be reused for initializing
            the current executor. This buffer will be checked for reuse if one argument name
            of the current executor is not found in `shared_arg_names`. The `NDArray` s are
            expected have default storage type.

        kwargs : Dict of str->shape
            Input shape dictionary, name->shape

        Returns
        -------
        executor : mxnet.Executor
            The generated executor
        """
    num_provided_arg_types = 0
    provided_arg_type_names = ctypes.POINTER(ctypes.c_char_p)()
    provided_arg_type_data = ctypes.POINTER(mx_uint)()
    if type_dict is not None:
        provided_arg_type_names = []
        provided_arg_type_data = []
        for k, v in type_dict.items():
            v = _numpy.dtype(v).type
            if v in _DTYPE_NP_TO_MX:
                provided_arg_type_names.append(k)
                provided_arg_type_data.append(_DTYPE_NP_TO_MX[v])
        num_provided_arg_types = mx_uint(len(provided_arg_type_names))
        provided_arg_type_names = c_str_array(provided_arg_type_names)
        provided_arg_type_data = c_array_buf(ctypes.c_int, array('i', provided_arg_type_data))
    num_provided_arg_stypes = 0
    provided_arg_stype_names = ctypes.POINTER(ctypes.c_char_p)()
    provided_arg_stype_data = ctypes.POINTER(mx_uint)()
    if stype_dict is not None:
        provided_arg_stype_names = []
        provided_arg_stype_data = []
        for k, v in stype_dict.items():
            if v in _STORAGE_TYPE_STR_TO_ID:
                provided_arg_stype_names.append(k)
                provided_arg_stype_data.append(_STORAGE_TYPE_STR_TO_ID[v])
        num_provided_arg_stypes = mx_uint(len(provided_arg_stype_names))
        provided_arg_stype_names = c_str_array(provided_arg_stype_names)
        provided_arg_stype_data = c_array_buf(ctypes.c_int, array('i', provided_arg_stype_data))
    provided_arg_shape_data = []
    provided_arg_shape_idx = [0]
    provided_arg_shape_names = []
    for k, v in kwargs.items():
        if isinstance(v, tuple):
            provided_arg_shape_names.append(k)
            provided_arg_shape_data.extend(v)
            provided_arg_shape_idx.append(len(provided_arg_shape_data))
    provided_req_type_list_len = 0
    provided_grad_req_types = ctypes.POINTER(ctypes.c_char_p)()
    provided_grad_req_names = ctypes.POINTER(ctypes.c_char_p)()
    if grad_req is not None:
        if isinstance(grad_req, string_types):
            provided_req_type_list_len = 0
            provided_grad_req_types = [grad_req]
        elif isinstance(grad_req, list):
            if len(grad_req) == 0:
                raise RuntimeError('grad_req in simple_bind cannot be an empty list')
            provided_grad_req_types = grad_req
            provided_req_type_list_len = len(provided_grad_req_types)
        elif isinstance(grad_req, dict):
            if len(grad_req) == 0:
                raise RuntimeError('grad_req in simple_bind cannot be an empty dict')
            provided_grad_req_names = []
            provided_grad_req_types = []
            for k, v in grad_req.items():
                provided_grad_req_names.append(k)
                provided_grad_req_types.append(v)
            provided_grad_req_names = c_str_array(provided_grad_req_names)
            provided_req_type_list_len = len(provided_grad_req_types)
        provided_grad_req_types = c_str_array(provided_grad_req_types)
    num_ctx_map_keys = mx_uint(0)
    ctx_map_keys = ctypes.POINTER(ctypes.c_char_p)()
    ctx_map_dev_types = ctypes.POINTER(ctypes.c_int)()
    ctx_map_dev_ids = ctypes.POINTER(ctypes.c_int)()
    if group2ctx is not None:
        ctx_map_keys = []
        ctx_map_dev_types = []
        ctx_map_dev_ids = []
        for key, val in group2ctx.items():
            ctx_map_keys.append(key)
            ctx_map_dev_types.append(val.device_typeid)
            ctx_map_dev_ids.append(val.device_id)
        num_ctx_map_keys = mx_uint(len(ctx_map_keys))
        ctx_map_keys = c_str_array(ctx_map_keys)
        ctx_map_dev_types = c_array(ctypes.c_int, array('i', ctx_map_dev_types))
        ctx_map_dev_ids = c_array(ctypes.c_int, array('i', ctx_map_dev_ids))
    shared_arg_name_list = []
    if shared_arg_names is not None:
        if not isinstance(shared_arg_names, list):
            raise ValueError('shared_arg_names in simple_bind must be a list or None')
        shared_arg_name_list = shared_arg_names
    if shared_buffer is None:
        shared_buffer_len = ctypes.c_int(-1)
        shared_buffer_names = ctypes.POINTER(ctypes.c_char_p)()
        shared_buffer_handles = ctypes.POINTER(NDArrayHandle)()
    else:
        if not isinstance(shared_buffer, dict):
            raise ValueError('shared_buffer in simple_bind must be dict or None')
        buffer_names = shared_buffer.keys()
        buffer_arrays = shared_buffer.values()
        for v in buffer_arrays:
            assert v.stype == 'default', 'shared_buffer is expected to only contain NDArrays with default storage'
        shared_buffer_names = c_str_array(buffer_names)
        shared_buffer_len = ctypes.c_int(len(buffer_arrays))
        shared_buffer_handles = c_handle_array(buffer_arrays)
    updated_shared_buffer_names = ctypes.POINTER(ctypes.c_char_p)()
    updated_shared_buffer_handles = ctypes.POINTER(NDArrayHandle)()
    shared_exec_handle = shared_exec.handle if shared_exec is not None else ExecutorHandle()
    exe_handle = ExecutorHandle()
    num_in_args = ctypes.c_uint()
    in_arg_handles = ctypes.POINTER(NDArrayHandle)()
    arg_grad_handles = ctypes.POINTER(NDArrayHandle)()
    num_aux_states = ctypes.c_uint()
    aux_state_handles = ctypes.POINTER(NDArrayHandle)()
    try:
        if _int64_enabled():
            check_call(_LIB.MXExecutorSimpleBindEx64(self.handle, ctypes.c_int(ctx.device_typeid), ctypes.c_int(ctx.device_id), num_ctx_map_keys, ctx_map_keys, ctx_map_dev_types, ctx_map_dev_ids, mx_uint(provided_req_type_list_len), provided_grad_req_names, provided_grad_req_types, mx_uint(len(provided_arg_shape_names)), c_str_array(provided_arg_shape_names), c_array_buf(mx_int64, array('q', provided_arg_shape_data)), c_array_buf(mx_uint, array('i', provided_arg_shape_idx)), num_provided_arg_types, provided_arg_type_names, provided_arg_type_data, num_provided_arg_stypes, provided_arg_stype_names, provided_arg_stype_data, mx_uint(len(shared_arg_name_list)), c_str_array(shared_arg_name_list), ctypes.byref(shared_buffer_len), shared_buffer_names, shared_buffer_handles, ctypes.byref(updated_shared_buffer_names), ctypes.byref(updated_shared_buffer_handles), ctypes.byref(num_in_args), ctypes.byref(in_arg_handles), ctypes.byref(arg_grad_handles), ctypes.byref(num_aux_states), ctypes.byref(aux_state_handles), shared_exec_handle, ctypes.byref(exe_handle)))
        else:
            check_call(_LIB.MXExecutorSimpleBindEx(self.handle, ctypes.c_int(ctx.device_typeid), ctypes.c_int(ctx.device_id), num_ctx_map_keys, ctx_map_keys, ctx_map_dev_types, ctx_map_dev_ids, mx_uint(provided_req_type_list_len), provided_grad_req_names, provided_grad_req_types, mx_uint(len(provided_arg_shape_names)), c_str_array(provided_arg_shape_names), c_array_buf(mx_int, array('I', provided_arg_shape_data)), c_array_buf(mx_uint, array('i', provided_arg_shape_idx)), num_provided_arg_types, provided_arg_type_names, provided_arg_type_data, num_provided_arg_stypes, provided_arg_stype_names, provided_arg_stype_data, mx_uint(len(shared_arg_name_list)), c_str_array(shared_arg_name_list), ctypes.byref(shared_buffer_len), shared_buffer_names, shared_buffer_handles, ctypes.byref(updated_shared_buffer_names), ctypes.byref(updated_shared_buffer_handles), ctypes.byref(num_in_args), ctypes.byref(in_arg_handles), ctypes.byref(arg_grad_handles), ctypes.byref(num_aux_states), ctypes.byref(aux_state_handles), shared_exec_handle, ctypes.byref(exe_handle)))
    except MXNetError as e:
        error_msg = 'simple_bind error. Arguments:\n'
        for k, v in kwargs.items():
            error_msg += '%s: %s\n' % (k, v)
        error_msg += '%s' % e
        raise RuntimeError(error_msg)
    if shared_buffer is not None:
        for i in range(shared_buffer_len.value):
            k = py_str(updated_shared_buffer_names[i])
            v = NDArray(NDArrayHandle(updated_shared_buffer_handles[i]))
            shared_buffer[k] = v
    arg_arrays = [_ndarray_cls(NDArrayHandle(in_arg_handles[i])) for i in range(num_in_args.value)]
    grad_arrays = [_ndarray_cls(NDArrayHandle(arg_grad_handles[i])) if arg_grad_handles[i] is not None else None for i in range(num_in_args.value)]
    aux_arrays = [_ndarray_cls(NDArrayHandle(aux_state_handles[i])) for i in range(num_aux_states.value)]
    executor = Executor(exe_handle, self, ctx, grad_req, group2ctx)
    executor.arg_arrays = arg_arrays
    executor.grad_arrays = grad_arrays
    executor.aux_arrays = aux_arrays
    return executor