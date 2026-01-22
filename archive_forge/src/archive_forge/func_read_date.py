import uuid
from datetime import datetime, time, date, timezone, timedelta
from decimal import Context
from .const import (
def read_date(data, writer_schema=None, reader_schema=None):
    return date.fromordinal(data + DAYS_SHIFT)