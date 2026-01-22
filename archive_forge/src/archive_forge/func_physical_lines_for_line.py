import math
import os
import re
import textwrap
from itertools import chain, groupby
from typing import (TYPE_CHECKING, Any, Dict, Generator, Iterable, List, Optional, Set, Tuple,
from docutils import nodes, writers
from docutils.nodes import Element, Text
from docutils.utils import column_width
from sphinx import addnodes
from sphinx.locale import _, admonitionlabels
from sphinx.util.docutils import SphinxTranslator
def physical_lines_for_line(self, line: List[Cell]) -> int:
    """For a given line, compute the number of physical lines it spans
        due to text wrapping.
        """
    physical_lines = 1
    for cell in line:
        physical_lines = max(physical_lines, len(cell.wrapped))
    return physical_lines