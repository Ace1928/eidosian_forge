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
def container_script_repr(container, imports, prefix, settings):
    result = []
    for i in container:
        result.append(pprint(i, imports, prefix, settings))
    if isinstance(container, list):
        d1, d2 = ('[', ']')
    elif isinstance(container, tuple):
        d1, d2 = ('(', ')')
    else:
        raise NotImplementedError
    rep = d1 + ','.join(result) + d2
    return rep