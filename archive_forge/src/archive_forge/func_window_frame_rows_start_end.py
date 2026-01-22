import datetime
import decimal
import json
import warnings
from importlib import import_module
import sqlparse
from django.conf import settings
from django.db import NotSupportedError, transaction
from django.db.backends import utils
from django.db.models.expressions import Col
from django.utils import timezone
from django.utils.deprecation import RemovedInDjango60Warning
from django.utils.encoding import force_str
def window_frame_rows_start_end(self, start=None, end=None):
    """
        Return SQL for start and end points in an OVER clause window frame.
        """
    if not self.connection.features.supports_over_clause:
        raise NotSupportedError('This backend does not support window expressions.')
    return (self.window_frame_start(start), self.window_frame_end(end))