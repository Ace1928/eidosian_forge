import sys
import textwrap
from difflib import unified_diff
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
from docutils import nodes
from docutils.nodes import Element, Node
from docutils.parsers.rst import directives
from docutils.statemachine import StringList
from sphinx import addnodes
from sphinx.config import Config
from sphinx.directives import optional_int
from sphinx.locale import __
from sphinx.util import logging, parselinenos
from sphinx.util.docutils import SphinxDirective
from sphinx.util.typing import OptionSpec
def dedent_filter(self, lines: List[str], location: Optional[Tuple[str, int]]=None) -> List[str]:
    if 'dedent' in self.options:
        return dedent_lines(lines, self.options.get('dedent'), location=location)
    else:
        return lines