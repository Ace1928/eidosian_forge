from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import textwrap
import unittest
from pasta.base import ast_utils
from pasta.base import scope
from pasta.base import test_utils
def test_if_nested_imports(self):
    source = textwrap.dedent('        if a:\n          import aaa\n        elif b:\n          import bbb\n        else:\n          import ccc\n        ')
    tree = ast.parse(source)
    nodes = tree.body
    node_aaa, node_bbb, node_ccc = ast_utils.find_nodes_by_type(tree, ast.alias)
    s = scope.analyze(tree)
    self.assertItemsEqual(s.names.keys(), {'aaa', 'bbb', 'ccc', 'a', 'b'})
    self.assertItemsEqual(s.external_references.keys(), {'aaa', 'bbb', 'ccc'})
    self.assertEqual(s.names['aaa'].definition, node_aaa)
    self.assertEqual(s.names['bbb'].definition, node_bbb)
    self.assertEqual(s.names['ccc'].definition, node_ccc)
    self.assertIsNone(s.names['a'].definition)
    self.assertIsNone(s.names['b'].definition)
    for ref in {'aaa', 'bbb', 'ccc'}:
        self.assertEqual(s.names[ref].reads, [], 'Expected no reads for %s' % ref)