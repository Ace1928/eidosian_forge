import functools
import zoneinfo
from contextlib import ContextDecorator
from datetime import datetime, timedelta, timezone, tzinfo
from asgiref.local import Local
from django.conf import settings
def template_localtime(value, use_tz=None):
    """
    Check if value is a datetime and converts it to local time if necessary.

    If use_tz is provided and is not None, that will force the value to
    be converted (or not), overriding the value of settings.USE_TZ.

    This function is designed for use by the template engine.
    """
    should_convert = isinstance(value, datetime) and (settings.USE_TZ if use_tz is None else use_tz) and (not is_naive(value)) and getattr(value, 'convert_to_local_time', True)
    return localtime(value) if should_convert else value