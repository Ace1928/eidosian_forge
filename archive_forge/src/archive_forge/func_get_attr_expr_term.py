import re
import sys
from collections import namedtuple
from typing import List, Dict, Any
from lark.tree import Meta
from lark.visitors import Transformer, Discard, _DiscardType, v_args
def get_attr_expr_term(self, args: List) -> str:
    return f'{args[0]}{args[1]}'