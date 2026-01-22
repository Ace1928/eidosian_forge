from bisect import bisect_left
from bisect import bisect_right
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
from inspect import isclass
import calendar
import collections
import datetime
import decimal
import hashlib
import itertools
import logging
import operator
import re
import socket
import struct
import sys
import threading
import time
import uuid
import warnings
def qualify_names(node):
    if isinstance(node, Expression):
        return node.__class__(qualify_names(node.lhs), node.op, qualify_names(node.rhs), node.flat)
    elif isinstance(node, ColumnBase):
        return QualifiedNames(node)
    return node