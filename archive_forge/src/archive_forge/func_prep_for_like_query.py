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
def prep_for_like_query(self, x):
    """Prepare a value for use in a LIKE query."""
    return str(x).replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')