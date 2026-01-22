import sys
from docutils import nodes, utils
from docutils.parsers.rst import Directive
from docutils.parsers.rst import states
from docutils.transforms import components
def field_marker(self, match, context, next_state):
    """Meta element."""
    node, blank_finish = self.parsemeta(match)
    self.parent += node
    return ([], next_state, [])