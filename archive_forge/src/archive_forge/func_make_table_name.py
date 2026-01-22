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
def make_table_name(self):
    if self.legacy_table_names:
        return re.sub('[^\\w]+', '_', self.name)
    return make_snake_case(self.model.__name__)