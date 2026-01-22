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
def register_aggregate(self, klass, name=None, num_params=-1):
    self._aggregates[name or klass.__name__.lower()] = (klass, num_params)
    if not self.is_closed():
        self._load_aggregates(self.connection())