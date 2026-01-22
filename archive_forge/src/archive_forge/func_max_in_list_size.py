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
def max_in_list_size(self):
    """
        Return the maximum number of items that can be passed in a single 'IN'
        list condition, or None if the backend does not impose a limit.
        """
    return None