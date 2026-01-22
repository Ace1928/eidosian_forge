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
def get_all_slots(class_):
    """
    Return a list of slot names for slots defined in `class_` and its
    superclasses.
    """
    all_slots = []
    parent_param_classes = [c for c in classlist(class_)[1:]]
    for c in parent_param_classes:
        if hasattr(c, '__slots__'):
            all_slots += c.__slots__
    return all_slots