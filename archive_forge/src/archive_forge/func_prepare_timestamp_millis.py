import datetime
import decimal
from io import BytesIO
import os
import time
from typing import Dict, Union
import uuid
from .const import (
def prepare_timestamp_millis(data, schema):
    """Converts datetime.datetime object to int timestamp with milliseconds"""
    if isinstance(data, datetime.datetime):
        if data.tzinfo is not None:
            delta = data - epoch
            return (delta.days * 24 * 3600 + delta.seconds) * MLS_PER_SECOND + int(delta.microseconds / 1000)
        if is_windows:
            delta = data - epoch_naive
            return (delta.days * 24 * 3600 + delta.seconds) * MLS_PER_SECOND + int(delta.microseconds / 1000)
        else:
            return int(time.mktime(data.timetuple())) * MLS_PER_SECOND + int(data.microsecond / 1000)
    else:
        return data