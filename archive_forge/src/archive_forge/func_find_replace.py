import os
import re
from docutils import nodes, transforms
from docutils.statemachine import StringList
from docutils.parsers.rst import Parser
from docutils.utils import new_document
from sphinx import addnodes
from .states import DummyStateMachine
def find_replace(self, node):
    """Try to find replace node for current node.

        Parameters
        ----------
        node : docutil node
            Node to find replacement for.

        Returns
        -------
        nodes : node or list of node
            The replacement nodes of current node.
            Returns None if no replacement can be found.
        """
    newnode = None
    if isinstance(node, nodes.Sequential):
        newnode = self.auto_toc_tree(node)
    elif isinstance(node, nodes.literal_block):
        newnode = self.auto_code_block(node)
    elif isinstance(node, nodes.literal):
        newnode = self.auto_inline_code(node)
    return newnode