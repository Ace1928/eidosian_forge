from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Type, Union, cast
from docutils import nodes
from docutils.nodes import Node
from docutils.parsers.rst.states import Inliner
from sphinx import addnodes
from sphinx.environment import BuildEnvironment
from sphinx.locale import __
from sphinx.util import logging
from sphinx.util.typing import TextlikeNode
def make_xrefs(self, rolename: str, domain: str, target: str, innernode: Type[TextlikeNode]=addnodes.literal_emphasis, contnode: Node=None, env: BuildEnvironment=None, inliner: Inliner=None, location: Node=None) -> List[Node]:
    return [self.make_xref(rolename, domain, target, innernode, contnode, env, inliner, location)]