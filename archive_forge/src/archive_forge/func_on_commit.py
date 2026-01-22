import _thread
import copy
import datetime
import logging
import threading
import time
import warnings
import zoneinfo
from collections import deque
from contextlib import contextmanager
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.db import DEFAULT_DB_ALIAS, DatabaseError, NotSupportedError
from django.db.backends import utils
from django.db.backends.base.validation import BaseDatabaseValidation
from django.db.backends.signals import connection_created
from django.db.backends.utils import debug_transaction
from django.db.transaction import TransactionManagementError
from django.db.utils import DatabaseErrorWrapper
from django.utils.asyncio import async_unsafe
from django.utils.functional import cached_property
def on_commit(self, func, robust=False):
    if not callable(func):
        raise TypeError("on_commit()'s callback must be a callable.")
    if self.in_atomic_block:
        self.run_on_commit.append((set(self.savepoint_ids), func, robust))
    elif not self.get_autocommit():
        raise TransactionManagementError('on_commit() cannot be used in manual transaction management')
    elif robust:
        try:
            func()
        except Exception as e:
            logger.error(f'Error calling {func.__qualname__} in on_commit() (%s).', e, exc_info=True)
    else:
        func()