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
def query_to_string(query):
    db = getattr(query, '_database', None)
    if db is not None:
        ctx = db.get_sql_context()
    else:
        ctx = Context()
    sql, params = ctx.sql(query).query()
    if not params:
        return sql
    param = ctx.state.param or '?'
    if param == '?':
        sql = sql.replace('?', '%s')
    return sql % tuple(map(_query_val_transform, params))