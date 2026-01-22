from __future__ import unicode_literals
import unittest
from cmakelang import configuration
from cmakelang import lex
from cmakelang import parse
from cmakelang.parse.printer import tree_string, test_string
from cmakelang.parse.common import NodeType
def assert_tree_type(test, nodes, tups, tree=None, history=None):
    """
  Check the output tree structure against that of expect_tree: a nested tuple
  tree.
  """
    if tree is None:
        tree = nodes
    if history is None:
        history = []
    for node, tup in overzip(nodes, tups):
        if isinstance(node, lex.Token):
            continue
        message = 'For node {} at\n {} within \n{}. If this is infact correct, copy-paste this:\n\n{}'.format(node, tree_string([node]), tree_string(tree, history), test_string(tree))
        test.assertIsNotNone(node, msg='Missing node ' + message)
        test.assertIsNotNone(tup, msg='Extra node ' + message)
        expect_type, expect_children = tup
        test.assertEqual(node.node_type, expect_type, msg='Expected type={} '.format(expect_type) + message)
        assert_tree_type(test, node.children, expect_children, tree, history + [node])