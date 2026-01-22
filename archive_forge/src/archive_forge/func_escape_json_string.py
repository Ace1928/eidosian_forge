import re
import math
from calendar import isleap, leapdays
from decimal import Decimal
from operator import attrgetter
from urllib.parse import urlsplit
from typing import Any, Iterator, List, Match, Optional, Union, SupportsFloat
def escape_json_string(s: str, escaped: bool=False) -> str:
    if escaped:
        s = s.replace('\\"', '"')
    else:
        s = s.replace('\\', '\\\\')
    s = s.replace('"', '\\"').replace('\x08', '\\b').replace('\r', '\\r').replace('\n', '\\n').replace('\t', '\\t').replace('\x0c', '\\f').replace('/', '\\/')
    return ''.join((f'\\u{ord(x):04X}' if 1 <= ord(x) <= 31 or 127 <= ord(x) <= 159 else x for x in s))