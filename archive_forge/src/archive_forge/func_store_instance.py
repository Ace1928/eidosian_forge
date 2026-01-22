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
def store_instance(self, instance, id_map):
    for field, attname in self.field_to_name:
        identity = field.rel_field.python_value(instance.__data__[attname])
        key = (field, identity)
        if self.is_backref:
            id_map[key] = instance
        else:
            id_map.setdefault(key, [])
            id_map[key].append(instance)