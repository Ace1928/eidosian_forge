from array import array
import ctypes
import logging
import contextlib
import numpy as np
from ... import symbol
from ...context import gpu
from ...symbol import Symbol
from ...module import BucketingModule
from ...symbol import contrib as symbol_contrib
from ... import ndarray
from ...ndarray import NDArray, _DTYPE_NP_TO_MX, _DTYPE_MX_TO_NP
from . import lists
from ...gluon import trainer
from ... import base
from ...base import c_str_array, SymbolHandle, check_call, _LIB, mx_uint, c_array_buf
from ... import optimizer as opt
from .loss_scaler import LossScaler
def convert_hybrid_block(block, target_dtype='float16', target_dtype_ops=None, fp32_ops=None, conditional_fp32_ops=None, excluded_sym_names=None, ctx=gpu(0), cast_optional_params=False):
    """Given a hybrid block/symbol block representing a FP32 model and a target_dtype,
    return a block with mixed precision support which can be used for inference use cases.

    Parameters
    ----------
    block : HybridBlock or SymbolBlock object
        FP32 HybridBlock or SymbolBlock object
    target_dtype : str or numpy
        currently only supports float16 and bfloat16. The target dtype indicates to add cast layers
        when possible so that lower precision computation can be leveraged.
    target_precision_ops : list of strs
        Override the list of operator names casted to target_dtype.
        If None, uses the framework's default list to be casted to FP32.
    conditional_fp32_ops : list of (str, str, list of str)
        Override the list of functions to be casted to FP32.
        The format of the list is
        (name of the function, name of the parameter,
         list of values of the parameter that make the operator to be casted to FP32
    excluded_sym_names : list of strs
        A list of strings that represent the names of symbols that users want to exclude
        from being quantized
    ctx : Context
        Context on which model parameters should live
    cast_optional_params : bool, default False
        Whether to cast the arg_params and aux_params that don't require to be in LP16
        because of a cast layer following it, but will reduce the computation and memory
        overhead of the model if casted.
    """
    from ...gluon import HybridBlock, SymbolBlock
    assert isinstance(block, HybridBlock), 'block input should be a HybridBlock'
    if not block._cached_graph:
        raise RuntimeError('Please first call block.hybridize() and then run forward with this block at least once before calling export.')
    inputs, sym = block._cached_graph
    input_names = []
    for inp in inputs:
        input_names.append(inp.name)
    converted_sym = convert_symbol(sym, target_dtype, target_dtype_ops, fp32_ops, conditional_fp32_ops, excluded_sym_names, data_names=input_names, cast_optional_params=cast_optional_params)
    arg_names = set(converted_sym.list_arguments())
    aux_names = set(converted_sym.list_auxiliary_states())
    arg_dict = {}
    attr_dict = converted_sym.attr_dict()
    for name, param in block.collect_params().items():
        if name in arg_names:
            arg_dict['arg:%s' % name] = param._reduce()
            if name in attr_dict and '__dtype__' in attr_dict[name]:
                if attr_dict[name]['__dtype__'] != '-1':
                    typ = _DTYPE_MX_TO_NP[int(attr_dict[name]['__dtype__'])]
                    if typ == bfloat16:
                        arg_dict['arg:%s' % name] = _cast_symbol_NDArray(arg_dict['arg:%s' % name], bfloat16)
                    else:
                        arg_dict['arg:%s' % name] = arg_dict['arg:%s' % name].astype(typ)
        else:
            assert name in aux_names
            arg_dict['aux:%s' % name] = param._reduce()
            if name in attr_dict and '__dtype__' in attr_dict[name]:
                if attr_dict[name]['__dtype__'] != '-1':
                    typ = _DTYPE_MX_TO_NP[int(attr_dict[name]['__dtype__'])]
                    if typ == bfloat16:
                        arg_dict['aux:%s' % name] = _cast_symbol_NDArray(arg_dict['aux:%s' % name], 'bfloat16')
                    else:
                        arg_dict['aux:%s' % name] = arg_dict['aux:%s' % name].astype(typ)
    ret = SymbolBlock(converted_sym, inputs)
    for key, param in ret.collect_params().items():
        arg_param_name = 'arg:%s' % key
        if arg_param_name in arg_dict and param.dtype != arg_dict[arg_param_name].dtype:
            param.cast(arg_dict[arg_param_name].dtype)
        aux_param_name = 'aux:%s' % key
        if aux_param_name in arg_dict and param.dtype != arg_dict[aux_param_name].dtype:
            param.cast(arg_dict[aux_param_name].dtype)
    ret.collect_params().load_dict(arg_dict, ctx=ctx)
    return ret