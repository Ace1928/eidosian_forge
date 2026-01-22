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
@cached_property
def pieces(self) -> List[range]:
    if not self.tree:
        return [range(i, i + 1) for i in range(1, len(self.lines) + 1)]
    return list(self._clean_pieces())