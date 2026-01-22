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
def get_views(self, schema=None):
    query = 'SELECT table_name, view_definition FROM information_schema.views WHERE table_schema = DATABASE() ORDER BY table_name'
    cursor = self.execute_sql(query)
    return [ViewMetadata(*row) for row in cursor.fetchall()]