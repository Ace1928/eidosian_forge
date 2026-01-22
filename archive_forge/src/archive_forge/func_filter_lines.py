import html.entities as htmlentitydefs
import re
import warnings
from ast import literal_eval
from collections import defaultdict
from enum import Enum
from io import StringIO
from typing import Any, NamedTuple
import networkx as nx
from networkx.exception import NetworkXError
from networkx.utils import open_file
def filter_lines(lines):
    if isinstance(lines, str):
        lines = decode_line(lines)
        lines = lines.splitlines()
        yield from lines
    else:
        for line in lines:
            line = decode_line(line)
            if line and line[-1] == '\n':
                line = line[:-1]
            if line.find('\n') != -1:
                raise NetworkXError('input line contains newline')
            yield line