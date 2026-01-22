import datetime
import decimal
import functools
import logging
import time
import warnings
from contextlib import contextmanager
from hashlib import md5
from django.apps import apps
from django.db import NotSupportedError
from django.utils.dateparse import parse_time
@contextmanager
def debug_transaction(connection, sql):
    start = time.monotonic()
    try:
        yield
    finally:
        if connection.queries_logged:
            stop = time.monotonic()
            duration = stop - start
            connection.queries_log.append({'sql': '%s' % sql, 'time': '%.3f' % duration})
            logger.debug('(%.3f) %s; args=%s; alias=%s', duration, sql, None, connection.alias, extra={'duration': duration, 'sql': sql, 'alias': connection.alias})