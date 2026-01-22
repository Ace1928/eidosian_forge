from __future__ import annotations
from contextlib import contextmanager
from enum import Enum
from functools import partial, wraps
from typing import Callable, List, Sequence, TypeVar
from .._C.libtriton.triton import ir
from . import semantic
def extern_elementwise(lib_name: str, lib_path: str, args: list, arg_type_symbol_dict: dict, is_pure: bool, _builder=None):
    """
        Dispatch an elementwise function to a library
        :param lib_name: the name of the library
        :param lib_path: the path of the library
        :param args: the arguments of the function
        :param arg_type_symbol_dict: the type of the arguments
        :param is_pure: whether the function is pure
        :param _builder: the builder
        :return: the return value of the function
    """
    dispatch_args = args.copy()
    all_scalar = True
    ret_shape = None
    arg_types = []
    for i in range(len(dispatch_args)):
        dispatch_args[i] = _to_tensor(dispatch_args[i], _builder)
        arg_types.append(dispatch_args[i].dtype)
        if dispatch_args[i].type.is_block():
            all_scalar = False
    if len(arg_types) > 0:
        arg_types = tuple(arg_types)
        arithmetic_check = True
        if arg_types in arg_type_symbol_dict:
            arithmetic_check = False
        broadcast_arg = dispatch_args[0]
        for i, item in enumerate(dispatch_args):
            _, broadcast_arg = semantic.binary_op_type_checking_impl(item, broadcast_arg, _builder, arithmetic_check=arithmetic_check)
        for i in range(len(dispatch_args)):
            dispatch_args[i], _ = semantic.binary_op_type_checking_impl(dispatch_args[i], broadcast_arg, _builder, arithmetic_check=arithmetic_check)
        if not all_scalar:
            ret_shape = broadcast_arg.shape
    func = getattr(_builder, 'create_extern_elementwise')
    return dispatch(func, lib_name, lib_path, dispatch_args, arg_type_symbol_dict, ret_shape, is_pure, _builder)