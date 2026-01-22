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
def variable_ranges(self) -> List[RangeInLine]:
    """
        A list of RangeInLines for each Variable that appears at least partially in this line.
        The data attribute of the range is a pair (variable, node) where node is the particular
        AST node from the list variable.nodes that corresponds to this range.
        """
    return [self.range_from_node(node, (variable, node)) for variable, node in self.frame_info.variables_by_lineno[self.lineno]]