from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import textwrap
import unittest
from pasta.base import ast_utils
from pasta.base import scope
from pasta.base import test_utils
@test_utils.requires_features('type_annotations')
def test_import_in_argument_type(self):
    source = textwrap.dedent('        import aaa\n        def foo(bar: aaa.Bar):\n          pass\n        ')
    tree = ast.parse(source)
    nodes = tree.body
    func = nodes[1]
    s = scope.analyze(tree)
    self.assertItemsEqual(s.names.keys(), {'aaa', 'foo'})
    self.assertItemsEqual(s.external_references.keys(), {'aaa'})
    self.assertItemsEqual(s.names['aaa'].reads, [func.args.args[0].annotation.value])