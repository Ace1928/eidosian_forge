import os
import pathlib
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple, Union
def materialize_lines(lines: List[str], indentation: int) -> str:
    output = ''
    new_line_with_indent = '\n' + ' ' * indentation
    for i, line in enumerate(lines):
        if i != 0:
            output += new_line_with_indent
        output += line.replace('\n', new_line_with_indent)
    return output