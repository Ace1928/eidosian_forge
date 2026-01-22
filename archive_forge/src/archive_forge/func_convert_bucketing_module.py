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
def convert_bucketing_module(bucketing_mod, target_dtype='float16', target_dtype_ops=None, fp32_ops=None, conditional_fp32_ops=None, excluded_sym_names=None, cast_optional_params=False):
    """Given a bucketing module cast the symbols associated with the BucketingModule
    and params if cast_optional_params is set.
    bucketing_mod : BucketingModule instance
    target_dtype : str
        Currently only supports float16. The target dtype indicates to add cast layers
        when possible so that lower precision computation can be leveraged.
    target_dtype_ops : list of strs
        Override the list of operator names casted to target_dtype.
        If None, uses the framework's default list to be casted to target dtype.
    fp32_ops : list of strs
        Override the lists of operator names casted to FP32.
        If None, uses the framework's default list to be casted to FP32.
    widest_dtype_ops : list of strs
        A list of op names provided by user which should run in widest precision among its inputs.
        If None, uses the framework's default list of widest_precision_ops.
    conditional_fp32_ops : list of (string, string, list of string)
        Override the list of operators to be casted to FP32.
        The format of the list is
        (name of the function, name of the parameter,
         list of values of the parameter that make the operator to be casted to
        fp32)
    excluded_sym_names : list of strs
        A list of strings that represent the names of symbols that users want to exclude
        from being executed in lower precision.
    cast_optional_params : bool, default False
        Whether to cast the arg_params and aux_params that don't require to be in LP16
        because of a cast layer following it, but will reduce the computation and memory
        overhead of the model if casted.
    """
    assert isinstance(bucketing_mod, BucketingModule), 'module should be instance of bucketing module'
    assert len(bucketing_mod._buckets) > 0, 'Bucketing Module should not be empty'
    sym_dict = {}
    assert bucketing_mod.params_initialized, 'bucketing_mod params should be initialized for mixed precision conversion'
    arg_params, aux_params = (bucketing_mod._curr_module._arg_params, bucketing_mod._curr_module._aux_params)
    for key, val in bucketing_mod._buckets.items():
        sym_dict[key], result_arg_params, result_aux_params = convert_model(val._symbol, arg_params, aux_params, target_dtype=target_dtype, target_dtype_ops=target_dtype_ops, fp32_ops=fp32_ops, conditional_fp32_ops=conditional_fp32_ops, excluded_sym_names=excluded_sym_names, cast_optional_params=cast_optional_params)
    result_mod = BucketingModule.load_dict(sym_dict, sym_gen=bucketing_mod._sym_gen, arg_params=result_arg_params, aux_params=result_aux_params, default_bucket_key=bucketing_mod._default_bucket_key, logger=bucketing_mod.logger, context=bucketing_mod._context, work_load_list=bucketing_mod._work_load_list, fixed_param_names=bucketing_mod._fixed_param_names, state_names=bucketing_mod._state_names, group2ctxs=bucketing_mod._group2ctxs, compression_params=bucketing_mod._compression_params)
    return result_mod