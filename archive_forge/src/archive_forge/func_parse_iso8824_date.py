import functools
import logging
import re
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from io import DEFAULT_BUFFER_SIZE, BytesIO
from os import SEEK_CUR
from typing import (
from .errors import (
def parse_iso8824_date(text: Optional[str]) -> Optional[datetime]:
    orgtext = text
    if text is None:
        return None
    if text[0].isdigit():
        text = 'D:' + text
    if text.endswith(('Z', 'z')):
        text += '0000'
    text = text.replace('z', '+').replace('Z', '+').replace("'", '')
    i = max(text.find('+'), text.find('-'))
    if i > 0 and i != len(text) - 5:
        text += '00'
    for f in ('D:%Y', 'D:%Y%m', 'D:%Y%m%d', 'D:%Y%m%d%H', 'D:%Y%m%d%H%M', 'D:%Y%m%d%H%M%S', 'D:%Y%m%d%H%M%S%z'):
        try:
            d = datetime.strptime(text, f)
        except ValueError:
            continue
        else:
            if text[-5:] == '+0000':
                d = d.replace(tzinfo=timezone.utc)
            return d
    raise ValueError(f'Can not convert date: {orgtext}')