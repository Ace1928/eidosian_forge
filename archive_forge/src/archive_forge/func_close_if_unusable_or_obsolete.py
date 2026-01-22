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
def close_if_unusable_or_obsolete(self):
    """
        Close the current connection if unrecoverable errors have occurred
        or if it outlived its maximum age.
        """
    if self.connection is not None:
        self.health_check_done = False
        if self.get_autocommit() != self.settings_dict['AUTOCOMMIT']:
            self.close()
            return
        if self.errors_occurred:
            if self.is_usable():
                self.errors_occurred = False
                self.health_check_done = True
            else:
                self.close()
                return
        if self.close_at is not None and time.monotonic() >= self.close_at:
            self.close()
            return