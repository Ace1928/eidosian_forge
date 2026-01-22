import re
import sys
from collections import namedtuple
from typing import List, Dict, Any
from lark.tree import Meta
from lark.visitors import Transformer, Discard, _DiscardType, v_args
def for_object_expr(self, args: List) -> str:
    args = self.strip_new_line_tokens(args)
    for_expr = ' '.join([str(arg) for arg in args[1:-1]])
    return f'{{{for_expr}}}'