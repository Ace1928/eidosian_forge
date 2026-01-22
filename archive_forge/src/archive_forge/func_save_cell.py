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
@register(CellType)
def save_cell(pickler, obj):
    try:
        f = obj.cell_contents
    except ValueError:
        logger.trace(pickler, 'Ce3: %s', obj)
        pickler.save_reduce(_create_cell, (_shims._CELL_EMPTY,), obj=obj)
        pickler.save_reduce(_shims._delattr, (obj, 'cell_contents'))
        pickler.write(POP)
        logger.trace(pickler, '# Ce3')
        return
    if is_dill(pickler, child=True):
        if id(f) in pickler._postproc:
            postproc = pickler._postproc[id(f)]
        else:
            postproc = next(iter(pickler._postproc.values()), None)
        if postproc is not None:
            logger.trace(pickler, 'Ce2: %s', obj)
            pickler.save_reduce(_create_cell, (_CELL_REF,), obj=obj)
            postproc.append((_shims._setattr, (obj, 'cell_contents', f)))
            logger.trace(pickler, '# Ce2')
            return
    logger.trace(pickler, 'Ce1: %s', obj)
    pickler.save_reduce(_create_cell, (f,), obj=obj)
    logger.trace(pickler, '# Ce1')
    return