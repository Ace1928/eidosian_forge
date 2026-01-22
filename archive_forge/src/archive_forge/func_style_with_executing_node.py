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
def style_with_executing_node(style, modifier):
    from pygments.styles import get_style_by_name
    if isinstance(style, str):
        style = get_style_by_name(style)

    class NewStyle(style):
        for_executing_node = True
        styles = {**style.styles, **{k.ExecutingNode: v + ' ' + modifier for k, v in style.styles.items()}}
    return NewStyle