import re
from copy import deepcopy
from typing import Any, Callable, List, Match, Optional, Pattern, Tuple, Union
from docutils import nodes
from docutils.nodes import TextElement
from sphinx import addnodes
from sphinx.config import Config
from sphinx.util import logging
def assert_end(self, *, allowSemicolon: bool=False) -> None:
    self.skip_ws()
    if allowSemicolon:
        if not self.eof and self.definition[self.pos:] != ';':
            self.fail('Expected end of definition or ;.')
    elif not self.eof:
        self.fail('Expected end of definition.')