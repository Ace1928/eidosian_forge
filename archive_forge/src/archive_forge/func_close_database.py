import os
import sys
import warnings
from datetime import datetime, timezone
from importlib import import_module
from typing import IO, TYPE_CHECKING, Any, List, Optional, cast
from kombu.utils.imports import symbol_by_name
from kombu.utils.objects import cached_property
from celery import _state, signals
from celery.exceptions import FixupWarning, ImproperlyConfigured
def close_database(self, **kwargs: Any) -> None:
    if not self.db_reuse_max:
        return self._close_database()
    if self._db_recycles >= self.db_reuse_max * 2:
        self._db_recycles = 0
        self._close_database()
    self._db_recycles += 1