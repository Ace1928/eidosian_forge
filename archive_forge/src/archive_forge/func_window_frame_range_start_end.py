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
def window_frame_range_start_end(self, start=None, end=None):
    start_, end_ = self.window_frame_rows_start_end(start, end)
    features = self.connection.features
    if features.only_supports_unbounded_with_preceding_and_following and (start and start < 0 or (end and end > 0)):
        raise NotSupportedError('%s only supports UNBOUNDED together with PRECEDING and FOLLOWING.' % self.connection.display_name)
    return (start_, end_)