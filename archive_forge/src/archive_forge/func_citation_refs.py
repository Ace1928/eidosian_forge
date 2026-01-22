from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, cast
from docutils import nodes
from docutils.nodes import Element
from sphinx.addnodes import pending_xref
from sphinx.domains import Domain
from sphinx.locale import __
from sphinx.transforms import SphinxTransform
from sphinx.util import logging
from sphinx.util.nodes import copy_source_info, make_refnode
@property
def citation_refs(self) -> Dict[str, Set[str]]:
    return self.data.setdefault('citation_refs', {})