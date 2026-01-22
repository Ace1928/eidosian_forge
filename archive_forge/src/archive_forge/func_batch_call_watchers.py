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
@contextmanager
def batch_call_watchers(parameterized):
    """
    Context manager to batch events to provide to Watchers on a
    parameterized object.  This context manager queues any events
    triggered by setting a parameter on the supplied parameterized
    object, saving them up to dispatch them all at once when the
    context manager exits.
    """
    BATCH_WATCH = parameterized.param._BATCH_WATCH
    parameterized.param._BATCH_WATCH = True
    try:
        yield
    finally:
        parameterized.param._BATCH_WATCH = BATCH_WATCH
        if not BATCH_WATCH:
            parameterized.param._batch_call_watchers()