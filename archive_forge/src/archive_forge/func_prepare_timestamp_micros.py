import datetime
import decimal
from io import BytesIO
import os
import time
from typing import Dict, Union
import uuid
from .const import (
def prepare_timestamp_micros(data, schema):
    """Converts datetime.datetime to int timestamp with microseconds"""
    if isinstance(data, datetime.datetime):
        if data.tzinfo is not None:
            delta = data - epoch
            return (delta.days * 24 * 3600 + delta.seconds) * MCS_PER_SECOND + delta.microseconds
        if is_windows:
            delta = data - epoch_naive
            return (delta.days * 24 * 3600 + delta.seconds) * MCS_PER_SECOND + delta.microseconds
        else:
            return int(time.mktime(data.timetuple())) * MCS_PER_SECOND + data.microsecond
    else:
        return data