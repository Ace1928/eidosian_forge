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
def force_new_dynamic_value(self_, name):
    """
        Force a new value to be generated for the dynamic attribute
        name, and return it.

        If name is not dynamic, its current value is returned
        (i.e. equivalent to getattr(name).
        """
    cls_or_slf = self_.self_or_cls
    param_obj = cls_or_slf.param.objects('existing').get(name)
    if not param_obj:
        return getattr(cls_or_slf, name)
    cls, slf = (None, None)
    if isinstance(cls_or_slf, type):
        cls = cls_or_slf
    else:
        slf = cls_or_slf
    if not hasattr(param_obj, '_force'):
        return param_obj.__get__(slf, cls)
    else:
        return param_obj._force(slf, cls)