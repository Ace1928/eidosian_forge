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
@_deprecated(extra_msg='Use instead `.param.method_dependencies`')
def params_depended_on(self_, *args, **kwargs):
    """
        Given the name of a method, returns a PInfo object for each dependency
        of this method. See help(PInfo) for the contents of these objects.

        By default intermediate dependencies on sub-objects are not
        returned as these are primarily useful for internal use to
        determine when a sub-object dependency has to be updated.

        .. deprecated: 2.0.0
            Use instead `.param.method_dependencies`
        """
    return self_.method_dependencies(*args, **kwargs)