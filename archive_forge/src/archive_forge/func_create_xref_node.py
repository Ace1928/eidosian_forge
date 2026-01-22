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
def create_xref_node(self) -> Tuple[List[Node], List[system_message]]:
    target = self.target
    title = self.title
    if self.lowercase:
        target = target.lower()
    if self.fix_parens:
        title, target = self.update_title_and_target(title, target)
    options = {'refdoc': self.env.docname, 'refdomain': self.refdomain, 'reftype': self.reftype, 'refexplicit': self.has_explicit_title, 'refwarn': self.warn_dangling}
    refnode = self.nodeclass(self.rawtext, **options)
    self.set_source_info(refnode)
    title, target = self.process_link(self.env, refnode, self.has_explicit_title, title, target)
    refnode['reftarget'] = target
    refnode += self.innernodeclass(self.rawtext, title, classes=self.classes)
    return self.result_nodes(self.inliner.document, self.env, refnode, is_ref=True)