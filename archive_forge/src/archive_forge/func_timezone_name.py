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
@cached_property
def timezone_name(self):
    """
        Name of the time zone of the database connection.
        """
    if not settings.USE_TZ:
        return settings.TIME_ZONE
    elif self.settings_dict['TIME_ZONE'] is None:
        return 'UTC'
    else:
        return self.settings_dict['TIME_ZONE']