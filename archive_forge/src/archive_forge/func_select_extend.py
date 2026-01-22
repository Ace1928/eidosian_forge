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
def select_extend(self, *columns):
    self._is_default = False
    fields = _normalize_model_select(columns)
    return super(ModelSelect, self).select_extend(*fields)