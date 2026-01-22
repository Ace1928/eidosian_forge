import re
import math
from calendar import isleap, leapdays
from decimal import Decimal
from operator import attrgetter
from urllib.parse import urlsplit
from typing import Any, Iterator, List, Match, Optional, Union, SupportsFloat
def iter_sequence(obj: Any) -> Iterator[Any]:
    if obj is None:
        return
    elif isinstance(obj, list):
        for item in obj:
            yield from iter_sequence(item)
    else:
        yield obj