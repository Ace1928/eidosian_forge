import re
from copy import deepcopy
from typing import Any, Callable, List, Match, Optional, Pattern, Tuple, Union
from docutils import nodes
from docutils.nodes import TextElement
from sphinx import addnodes
from sphinx.config import Config
from sphinx.util import logging
def skip_word_and_ws(self, word: str) -> bool:
    if self.skip_word(word):
        self.skip_ws()
        return True
    return False