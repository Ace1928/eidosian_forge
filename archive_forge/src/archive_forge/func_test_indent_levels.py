from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import difflib
import itertools
import os.path
from six import with_metaclass
import sys
import textwrap
import unittest
import pasta
from pasta.base import annotate
from pasta.base import ast_utils
from pasta.base import codegen
from pasta.base import formatting as fmt
from pasta.base import test_utils
def test_indent_levels(self):
    src = textwrap.dedent("        foo('begin')\n        if a:\n          foo('a1')\n          if b:\n            foo('b1')\n            if c:\n              foo('c1')\n            foo('b2')\n          foo('a2')\n        foo('end')\n        ")
    t = pasta.parse(src)
    call_nodes = ast_utils.find_nodes_by_type(t, (ast.Call,))
    call_nodes.sort(key=lambda node: node.lineno)
    begin, a1, b1, c1, b2, a2, end = call_nodes
    self.assertEqual('', fmt.get(begin, 'indent'))
    self.assertEqual('  ', fmt.get(a1, 'indent'))
    self.assertEqual('    ', fmt.get(b1, 'indent'))
    self.assertEqual('      ', fmt.get(c1, 'indent'))
    self.assertEqual('    ', fmt.get(b2, 'indent'))
    self.assertEqual('  ', fmt.get(a2, 'indent'))
    self.assertEqual('', fmt.get(end, 'indent'))