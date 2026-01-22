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
def convert_dict_to_node(self, qdict):
    accum = []
    joins = []
    fks = (ForeignKeyField, BackrefAccessor)
    for key, value in sorted(qdict.items()):
        curr = self.model
        if '__' in key and key.rsplit('__', 1)[1] in DJANGO_MAP:
            key, op = key.rsplit('__', 1)
            op = DJANGO_MAP[op]
        elif value is None:
            op = DJANGO_MAP['is']
        else:
            op = DJANGO_MAP['eq']
        if '__' not in key:
            model_attr = getattr(curr, key)
        else:
            for piece in key.split('__'):
                for dest, attr, _, _ in self._joins.get(curr, ()):
                    try:
                        model_attr = getattr(curr, piece, None)
                    except:
                        pass
                    if attr == piece or (isinstance(dest, ModelAlias) and dest.alias == piece):
                        curr = dest
                        break
                else:
                    model_attr = getattr(curr, piece)
                    if value is not None and isinstance(model_attr, fks):
                        curr = model_attr.rel_model
                        joins.append(model_attr)
        accum.append(op(model_attr, value))
    return (accum, joins)