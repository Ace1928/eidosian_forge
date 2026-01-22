import sys
from os.path import splitext
from docutils import parsers, nodes
from sphinx import addnodes
from commonmark import Parser
from warnings import warn
def depart_link(self, mdnode):
    if isinstance(self.current_node.parent, addnodes.pending_xref):
        self.current_node = self.current_node.parent.parent
    else:
        self.current_node = self.current_node.parent