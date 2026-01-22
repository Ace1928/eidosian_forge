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
def variables_by_lineno(self) -> Mapping[int, List[Tuple[Variable, ast.AST]]]:
    """
        A mapping from 1-based line numbers to lists of pairs:
            - A Variable object
            - A specific AST node from the variable's .nodes list that's
                in the line at that line number.
        """
    result = defaultdict(list)
    for var in self.variables:
        for node in var.nodes:
            for lineno in range(*self.source.line_range(node)):
                result[lineno].append((var, node))
    return result