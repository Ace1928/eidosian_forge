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
def register_window_function(self, klass, name=None, num_params=-1):
    name = name or klass.__name__.lower()
    self._window_functions[name] = (klass, num_params)
    if not self.is_closed():
        self._load_window_functions(self.connection())