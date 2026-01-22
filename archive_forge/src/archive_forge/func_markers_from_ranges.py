import ast
import html
import os
import sys
from collections import defaultdict, Counter
from enum import Enum
from textwrap import dedent
from types import FrameType, CodeType, TracebackType
from typing import (
from typing import Mapping
import executing
from asttokens.util import Token
from executing import only
from pure_eval import Evaluator, is_expression_interesting
from stack_data.utils import (
def markers_from_ranges(ranges: Iterable[RangeInLine], converter: Callable[[RangeInLine], Optional[Tuple[str, str]]]) -> List[MarkerInLine]:
    """
    Helper to create MarkerInLines given some RangeInLines.
    converter should be a function accepting a RangeInLine returning
    either None (which is ignored) or a pair of strings which
    are used to create two markers included in the returned list.
    """
    markers = []
    for rang in ranges:
        converted = converter(rang)
        if converted is None:
            continue
        start_string, end_string = converted
        if not (isinstance(start_string, str) and isinstance(end_string, str)):
            raise TypeError('converter should return None or a pair of strings')
        markers += [MarkerInLine(position=rang.start, is_start=True, string=start_string), MarkerInLine(position=rang.end, is_start=False, string=end_string)]
    return markers