import sys
from os.path import splitext
from docutils import parsers, nodes
from sphinx import addnodes
from commonmark import Parser
from warnings import warn
def visit_thematic_break(self, _):
    self.current_node.append(nodes.transition())