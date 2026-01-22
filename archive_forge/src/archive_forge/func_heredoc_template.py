import re
import sys
from collections import namedtuple
from typing import List, Dict, Any
from lark.tree import Meta
from lark.visitors import Transformer, Discard, _DiscardType, v_args
def heredoc_template(self, args: List) -> str:
    match = HEREDOC_PATTERN.match(str(args[0]))
    if not match:
        raise RuntimeError(f'Invalid Heredoc token: {args[0]}')
    trim_chars = '\n\t '
    return f'"{match.group(2).rstrip(trim_chars)}"'