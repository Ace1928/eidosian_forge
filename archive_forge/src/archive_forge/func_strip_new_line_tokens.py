import re
import sys
from collections import namedtuple
from typing import List, Dict, Any
from lark.tree import Meta
from lark.visitors import Transformer, Discard, _DiscardType, v_args
def strip_new_line_tokens(self, args: List) -> List:
    """
        Remove new line and Discard tokens.
        The parser will sometimes include these in the tree so we need to strip them out here
        """
    return [arg for arg in args if arg != '\n' and arg is not Discard]