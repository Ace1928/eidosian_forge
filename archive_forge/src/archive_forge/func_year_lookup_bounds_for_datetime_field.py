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
def year_lookup_bounds_for_datetime_field(self, value, iso_year=False):
    """
        Return a two-elements list with the lower and upper bound to be used
        with a BETWEEN operator to query a DateTimeField value using a year
        lookup.

        `value` is an int, containing the looked-up year.
        If `iso_year` is True, return bounds for ISO-8601 week-numbering years.
        """
    if iso_year:
        first = datetime.datetime.fromisocalendar(value, 1, 1)
        second = datetime.datetime.fromisocalendar(value + 1, 1, 1) - datetime.timedelta(microseconds=1)
    else:
        first = datetime.datetime(value, 1, 1)
        second = datetime.datetime(value, 12, 31, 23, 59, 59, 999999)
    if settings.USE_TZ:
        tz = timezone.get_current_timezone()
        first = timezone.make_aware(first, tz)
        second = timezone.make_aware(second, tz)
    first = self.adapt_datetimefield_value(first)
    second = self.adapt_datetimefield_value(second)
    return [first, second]