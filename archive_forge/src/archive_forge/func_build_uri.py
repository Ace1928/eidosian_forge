import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type
import docutils.parsers.rst.directives
import docutils.parsers.rst.roles
import docutils.parsers.rst.states
from docutils import nodes, utils
from docutils.nodes import Element, Node, TextElement, system_message
from sphinx import addnodes
from sphinx.locale import _, __
from sphinx.util import ws_re
from sphinx.util.docutils import ReferenceRole, SphinxRole
from sphinx.util.typing import RoleFunction
def build_uri(self) -> str:
    base_url = self.inliner.document.settings.rfc_base_url
    ret = self.target.split('#', 1)
    if len(ret) == 2:
        return base_url + self.inliner.rfc_url % int(ret[0]) + '#' + ret[1]
    else:
        return base_url + self.inliner.rfc_url % int(ret[0])