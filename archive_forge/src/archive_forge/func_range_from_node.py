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
def range_from_node(self, node: ast.AST, data: Any, common_indent: int=0) -> Optional[RangeInLine]:
    """
        If the given node overlaps with this line, return a RangeInLine
        with the correct start and end and the given data.
        Otherwise, return None.
        """
    atext = self.frame_info.source.asttext()
    (start, range_start), (end, range_end) = atext.get_text_positions(node, padded=False)
    if not start <= self.lineno <= end:
        return None
    if start != self.lineno:
        range_start = common_indent
    if end != self.lineno:
        range_end = len(self.text)
    if range_start == range_end == 0:
        return None
    return RangeInLine(range_start, range_end, data)