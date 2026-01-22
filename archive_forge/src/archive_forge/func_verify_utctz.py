from abc import ABC, abstractmethod
import calendar
from collections import deque
from datetime import datetime, timedelta, tzinfo
from string import digits
import re
import time
import warnings
from git.util import IterableList, IterableObj, Actor
from typing import (
from git.types import Has_id_attribute, Literal  # , _T
def verify_utctz(offset: str) -> str:
    """
    :raise ValueError: If offset is incorrect

    :return: offset
    """
    fmt_exc = ValueError('Invalid timezone offset format: %s' % offset)
    if len(offset) != 5:
        raise fmt_exc
    if offset[0] not in '+-':
        raise fmt_exc
    if offset[1] not in digits or offset[2] not in digits or offset[3] not in digits or (offset[4] not in digits):
        raise fmt_exc
    return offset