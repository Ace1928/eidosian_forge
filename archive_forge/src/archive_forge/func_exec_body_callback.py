import re
import sys
import copy
import types
import inspect
import keyword
import builtins
import functools
import itertools
import abc
import _thread
from types import FunctionType, GenericAlias
def exec_body_callback(ns):
    ns.update(namespace)
    ns.update(defaults)
    ns['__annotations__'] = annotations