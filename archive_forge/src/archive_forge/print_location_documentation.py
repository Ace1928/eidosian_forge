import re
from typing import Optional, Tuple, cast
from .ast import Location
from .location import SourceLocation, get_location
from .source import Source
Print lines specified like this: ("prefix", "string")