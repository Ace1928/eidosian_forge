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
def remove_ref(self, field):
    rel = field.rel_model
    del self.refs[field]
    self.model_refs[rel].remove(field)
    del rel._meta.backrefs[field]
    rel._meta.model_backrefs[self.model].remove(field)