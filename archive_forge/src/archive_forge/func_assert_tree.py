from __future__ import unicode_literals
import contextlib
import logging
import unittest
import sys
from cmakelang.format import __main__
from cmakelang import configuration
from cmakelang import lex
from cmakelang import parse
from cmakelang.format import formatter
from cmakelang.parse.common import NodeType
def assert_tree(test, nodes, tups, tree=None, history=None):
    if tree is None:
        tree = nodes
    if history is None:
        history = []
    for node, tup in overzip(nodes, tups):
        if isinstance(node, lex.Token):
            continue
        subhistory = history + [node]
        message = ' for node {} at\n {} \n\n\nIf the actual result is expected, then update the test with this:\n# pylint: disable=bad-continuation\n# noqa: E122\n{}'.format(node, formatter.tree_string(tree, subhistory), formatter.test_string(tree))
        test.assertIsNotNone(node, msg='Missing node' + message)
        test.assertIsNotNone(tup, msg='Extra node' + message)
        if len(tup) == 6:
            ntype, wrap, row, col, colextent, expect_children = tup
            test.assertEqual(node.passno, wrap, msg='Expected wrap={}'.format(wrap) + message)
        else:
            ntype, row, col, colextent, expect_children = tup
        test.assertEqual(node.node_type, ntype, msg='Expected type={}'.format(ntype) + message)
        test.assertEqual(node.position[0], row, msg='Expected row={}'.format(row) + message)
        test.assertEqual(node.position[1], col, msg='Expected col={}'.format(col) + message)
        test.assertEqual(node.colextent, colextent, msg='Expected colextent={}'.format(colextent) + message)
        assert_tree(test, node.children, expect_children, tree, subhistory)