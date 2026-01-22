import os
import re
from docutils import nodes, transforms
from docutils.statemachine import StringList
from docutils.parsers.rst import Parser
from docutils.utils import new_document
from sphinx import addnodes
from .states import DummyStateMachine
def auto_toc_tree(self, node):
    """Try to convert a list block to toctree in rst.

        This function detects if the matches the condition and return
        a converted toc tree node. The matching condition:
        The list only contains one level, and only contains references

        Parameters
        ----------
        node: nodes.Sequential
            A list node in the doctree

        Returns
        -------
        tocnode: docutils node
            The converted toc tree node, None if conversion is not possible.
        """
    if not self.config['enable_auto_toc_tree']:
        return None
    sec = self.config['auto_toc_tree_section']
    if sec is not None:
        if node.parent is None:
            return None
        title = None
        if isinstance(node.parent, nodes.section):
            child = node.parent.first_child_matching_class(nodes.title)
            if child is not None:
                title = node.parent.children[child]
        elif isinstance(node.parent, nodes.paragraph):
            child = node.parent.parent.first_child_matching_class(nodes.title)
            if child is not None:
                title = node.parent.parent.children[child]
        if not title:
            return None
        if title.astext().strip() != sec:
            return None
    numbered = None
    if isinstance(node, nodes.bullet_list):
        numbered = 0
    elif isinstance(node, nodes.enumerated_list):
        numbered = 1
    if numbered is None:
        return None
    refs = []
    for nd in node.children[:]:
        assert isinstance(nd, nodes.list_item)
        if len(nd.children) != 1:
            return None
        par = nd.children[0]
        if not isinstance(par, nodes.paragraph):
            return None
        if len(par.children) != 1:
            return None
        ref = par.children[0]
        if isinstance(ref, addnodes.pending_xref):
            ref = ref.children[0]
        if not isinstance(ref, nodes.reference):
            return None
        title, uri, docpath = self.parse_ref(ref)
        if title is None or uri.startswith('#'):
            return None
        if docpath:
            refs.append((title, docpath))
        else:
            refs.append((title, uri))
    self.state_machine.reset(self.document, node.parent, self.current_level)
    return self.state_machine.run_directive('toctree', options={'maxdepth': self.config['auto_toc_maxdepth'], 'numbered': numbered}, content=['%s <%s>' % (k, v) for k, v in refs])