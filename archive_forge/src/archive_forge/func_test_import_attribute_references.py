from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import textwrap
import unittest
from pasta.base import ast_utils
from pasta.base import scope
from pasta.base import test_utils
def test_import_attribute_references(self):
    source = textwrap.dedent('        import aaa.bbb.ccc, ddd.eee\n        aaa.x()\n        aaa.bbb.y()\n        aaa.bbb.ccc.z()\n        ')
    tree = ast.parse(source)
    nodes = tree.body
    call1 = nodes[1].value.func.value
    call2 = nodes[2].value.func.value
    call3 = nodes[3].value.func.value
    s = scope.analyze(tree)
    self.assertItemsEqual(s.names.keys(), {'aaa', 'ddd'})
    self.assertItemsEqual(s.external_references.keys(), {'aaa', 'aaa.bbb', 'aaa.bbb.ccc', 'ddd', 'ddd.eee'})
    self.assertItemsEqual(s.names['aaa'].reads, [call1, call2.value, call3.value.value])
    self.assertItemsEqual(s.names['aaa'].attrs['bbb'].reads, [call2, call3.value])
    self.assertItemsEqual(s.names['aaa'].attrs['bbb'].attrs['ccc'].reads, [call3])