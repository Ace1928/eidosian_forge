import datetime
from typing import Protocol, Tuple, Type, Union
def time_format_fn(sequence: str) -> datetime.time:
    return datetime.datetime.strptime(sequence, '%H:%M:%S').time()