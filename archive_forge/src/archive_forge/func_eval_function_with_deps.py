import asyncio
import copy
import datetime as dt
import html
import inspect
import logging
import numbers
import operator
import random
import re
import types
import typing
import warnings
from collections import defaultdict, namedtuple, OrderedDict
from functools import partial, wraps, reduce
from html import escape
from itertools import chain
from operator import itemgetter, attrgetter
from types import FunctionType, MethodType
from contextlib import contextmanager
from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL
from ._utils import (
from inspect import getfullargspec
def eval_function_with_deps(function):
    """Evaluates a function after resolving its dependencies.

    Calls and returns a function after resolving any dependencies
    stored on the _dinfo attribute and passing the resolved values
    as arguments.
    """
    args, kwargs = ((), {})
    if hasattr(function, '_dinfo'):
        arg_deps = function._dinfo['dependencies']
        kw_deps = function._dinfo.get('kw', {})
        if kw_deps or any((isinstance(d, Parameter) for d in arg_deps)):
            args = (getattr(dep.owner, dep.name) for dep in arg_deps)
            kwargs = {n: getattr(dep.owner, dep.name) for n, dep in kw_deps.items()}
    return function(*args, **kwargs)