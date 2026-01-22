import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, cast
from docutils import nodes
from docutils.nodes import Element, Node
from sphinx import addnodes
from sphinx.addnodes import pending_xref
from sphinx.application import Sphinx
from sphinx.domains import Domain
from sphinx.errors import NoUri
from sphinx.locale import __
from sphinx.transforms import SphinxTransform
from sphinx.util import logging
from sphinx.util.docutils import SphinxTranslator
from sphinx.util.nodes import find_pending_xref_condition, process_only_nodes
def matches_ignore(entry_type: str, entry_target: str) -> bool:
    for ignore_type, ignore_target in self.config.nitpick_ignore_regex:
        if re.fullmatch(ignore_type, entry_type) and re.fullmatch(ignore_target, entry_target):
            return True
    return False