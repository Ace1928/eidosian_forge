import datetime
from typing import Protocol, Tuple, Type, Union
def date_format_fn(sequence: str) -> datetime.date:
    return datetime.datetime.strptime(sequence, '%Y-%m-%d').date()