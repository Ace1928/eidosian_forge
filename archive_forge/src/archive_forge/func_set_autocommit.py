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
def set_autocommit(self, autocommit, force_begin_transaction_with_broken_autocommit=False):
    """
        Enable or disable autocommit.

        The usual way to start a transaction is to turn autocommit off.
        SQLite does not properly start a transaction when disabling
        autocommit. To avoid this buggy behavior and to actually enter a new
        transaction, an explicit BEGIN is required. Using
        force_begin_transaction_with_broken_autocommit=True will issue an
        explicit BEGIN with SQLite. This option will be ignored for other
        backends.
        """
    self.validate_no_atomic_block()
    self.close_if_health_check_failed()
    self.ensure_connection()
    start_transaction_under_autocommit = force_begin_transaction_with_broken_autocommit and (not autocommit) and hasattr(self, '_start_transaction_under_autocommit')
    if start_transaction_under_autocommit:
        self._start_transaction_under_autocommit()
    elif autocommit:
        self._set_autocommit(autocommit)
    else:
        with debug_transaction(self, 'BEGIN'):
            self._set_autocommit(autocommit)
    self.autocommit = autocommit
    if autocommit and self.run_commit_hooks_on_set_autocommit_on:
        self.run_and_clear_commit_hooks()
        self.run_commit_hooks_on_set_autocommit_on = False