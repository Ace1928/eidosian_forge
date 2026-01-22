from __future__ import annotations
import logging
import re
from typing import TYPE_CHECKING, Any, List, Optional, Pattern
from urllib.parse import urlparse
import numpy as np
def escape_symbol(match: re.Match) -> str:
    value = match.group(0)
    return f'\\{value}'