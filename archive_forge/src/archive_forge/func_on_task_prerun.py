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
def on_task_prerun(self, sender: 'Task', **kwargs: Any) -> None:
    """Called before every task."""
    if not getattr(sender.request, 'is_eager', False):
        self.close_database()