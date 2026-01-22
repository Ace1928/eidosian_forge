import re
import math
from calendar import isleap, leapdays
from decimal import Decimal
from operator import attrgetter
from urllib.parse import urlsplit
from typing import Any, Iterator, List, Match, Optional, Union, SupportsFloat
def is_xml_codepoint(cp: int) -> bool:
    return cp in (9, 10, 13) or 32 <= cp <= 55295 or 57344 <= cp <= 65533 or (65536 <= cp <= 1114111)