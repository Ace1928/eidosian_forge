import uuid
from datetime import datetime, time, date, timezone, timedelta
from decimal import Context
from .const import (
def read_local_timestamp_millis(data: int, writer_schema=None, reader_schema=None) -> datetime:
    return epoch_naive + timedelta(microseconds=data * 1000)