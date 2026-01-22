from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, cast
from docutils import nodes
from docutils.nodes import Node
from sphinx import addnodes
from sphinx.domains import Domain
from sphinx.locale import _
from sphinx.util.docutils import SphinxDirective
from sphinx.util.typing import OptionSpec
def note_changeset(self, node: addnodes.versionmodified) -> None:
    version = node['version']
    module = self.env.ref_context.get('py:module')
    objname = self.env.temp_data.get('object')
    changeset = ChangeSet(node['type'], self.env.docname, node.line, module, objname, node.astext())
    self.changesets.setdefault(version, []).append(changeset)