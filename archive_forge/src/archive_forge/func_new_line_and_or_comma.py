import re
import sys
from collections import namedtuple
from typing import List, Dict, Any
from lark.tree import Meta
from lark.visitors import Transformer, Discard, _DiscardType, v_args
def new_line_and_or_comma(self, args: List) -> _DiscardType:
    return Discard