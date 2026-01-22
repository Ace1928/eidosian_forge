import contextlib
import functools
import re
import textwrap
from typing import Iterable, List, Sequence, Tuple, Type
from typing_extensions import Literal, get_args, get_origin
from . import _resolver
def remove_single_line_breaks(helptext: str) -> str:
    lines = helptext.split('\n')
    output_parts: List[str] = []
    for line in lines:
        line = line.rstrip()
        if len(line) == 0:
            prev_is_break = len(output_parts) >= 1 and output_parts[-1] == '\n'
            if not prev_is_break:
                output_parts.append('\n')
            output_parts.append('\n')
        else:
            if not line[0].isalpha():
                output_parts.append('\n')
            prev_is_break = len(output_parts) >= 1 and output_parts[-1] == '\n'
            if len(output_parts) >= 1 and (not prev_is_break):
                output_parts.append(' ')
            output_parts.append(line)
    return ''.join(output_parts).rstrip()