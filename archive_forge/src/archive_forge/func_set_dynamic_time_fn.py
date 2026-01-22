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
def set_dynamic_time_fn(self_, time_fn, sublistattr=None):
    """
        Set time_fn for all Dynamic Parameters of this class or
        instance object that are currently being dynamically
        generated.

        Additionally, sets _Dynamic_time_fn=time_fn on this class or
        instance object, so that any future changes to Dynamic
        Parmeters can inherit time_fn (e.g. if a Number is changed
        from a float to a number generator, the number generator will
        inherit time_fn).

        If specified, sublistattr is the name of an attribute of this
        class or instance that contains an iterable collection of
        subobjects on which set_dynamic_time_fn should be called.  If
        the attribute sublistattr is present on any of the subobjects,
        set_dynamic_time_fn() will be called for those, too.
        """
    self_or_cls = self_.self_or_cls
    self_or_cls._Dynamic_time_fn = time_fn
    if isinstance(self_or_cls, type):
        a = (None, self_or_cls)
    else:
        a = (self_or_cls,)
    for n, p in self_or_cls.param.objects('existing').items():
        if hasattr(p, '_value_is_dynamic'):
            if p._value_is_dynamic(*a):
                g = self_or_cls.param.get_value_generator(n)
                g._Dynamic_time_fn = time_fn
    if sublistattr:
        try:
            sublist = getattr(self_or_cls, sublistattr)
        except AttributeError:
            sublist = []
        for obj in sublist:
            obj.param.set_dynamic_time_fn(time_fn, sublistattr)