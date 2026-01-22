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
def end_state(self, wrap: bool=True, end: List[str]=[''], first: str=None) -> None:
    content = self.states.pop()
    maxindent = sum(self.stateindent)
    indent = self.stateindent.pop()
    result: List[Tuple[int, List[str]]] = []
    toformat: List[str] = []

    def do_format() -> None:
        if not toformat:
            return
        if wrap:
            res = my_wrap(''.join(toformat), width=MAXWIDTH - maxindent)
        else:
            res = ''.join(toformat).splitlines()
        if end:
            res += end
        result.append((indent, res))
    for itemindent, item in content:
        if itemindent == -1:
            toformat.append(item)
        else:
            do_format()
            result.append((indent + itemindent, item))
            toformat = []
    do_format()
    if first is not None and result:
        newindent = result[0][0] - indent
        if result[0][1] == ['']:
            result.insert(0, (newindent, [first]))
        else:
            text = first + result[0][1].pop(0)
            result.insert(0, (newindent, [text]))
    self.states[-1].extend(result)