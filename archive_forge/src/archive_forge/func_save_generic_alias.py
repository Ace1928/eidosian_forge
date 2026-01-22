from __future__ import annotations
import warnings
from .logger import adapter as logger
from .logger import trace as _trace
import os
import sys
import builtins as __builtin__
from pickle import _Pickler as StockPickler, Unpickler as StockUnpickler
from pickle import GLOBAL, POP
from _thread import LockType
from _thread import RLock as RLockType
from types import CodeType, FunctionType, MethodType, GeneratorType, \
from types import MappingProxyType as DictProxyType, new_class
from pickle import DEFAULT_PROTOCOL, HIGHEST_PROTOCOL, PickleError, PicklingError, UnpicklingError
import __main__ as _main_module
import marshal
import gc
import abc
import dataclasses
from weakref import ReferenceType, ProxyType, CallableProxyType
from collections import OrderedDict
from enum import Enum, EnumMeta
from functools import partial
from operator import itemgetter, attrgetter
import importlib.machinery
from types import GetSetDescriptorType, ClassMethodDescriptorType, \
from io import BytesIO as StringIO
from socket import socket as SocketType
from multiprocessing.reduction import _reduce_socket as reduce_socket
import inspect
import typing
from . import _shims
from ._shims import Reduce, Getattr
@register(typing._GenericAlias)
def save_generic_alias(pickler, obj):
    args = obj.__args__
    if type(obj.__reduce__()) is str:
        logger.trace(pickler, 'Ga0: %s', obj)
        StockPickler.save_global(pickler, obj, name=obj.__reduce__())
        logger.trace(pickler, '# Ga0')
    elif obj.__origin__ is tuple and (not args or args == ((),)):
        logger.trace(pickler, 'Ga1: %s', obj)
        pickler.save_reduce(_create_typing_tuple, (args,), obj=obj)
        logger.trace(pickler, '# Ga1')
    else:
        logger.trace(pickler, 'Ga2: %s', obj)
        StockPickler.save_reduce(pickler, *obj.__reduce__(), obj=obj)
        logger.trace(pickler, '# Ga2')
    return