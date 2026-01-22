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
def test_indent_depths(self):
    template = 'if a:\n{first}if b:\n{first}{second}foo()\n'
    indents = (' ', ' ' * 2, ' ' * 4, ' ' * 8, '\t', '\t' * 2)
    for first, second in itertools.product(indents, indents):
        src = template.format(first=first, second=second)
        t = pasta.parse(src)
        outer_if_node = t.body[0]
        inner_if_node = outer_if_node.body[0]
        call_node = inner_if_node.body[0]
        self.assertEqual('', fmt.get(outer_if_node, 'indent'))
        self.assertEqual('', fmt.get(outer_if_node, 'indent_diff'))
        self.assertEqual(first, fmt.get(inner_if_node, 'indent'))
        self.assertEqual(first, fmt.get(inner_if_node, 'indent_diff'))
        self.assertEqual(first + second, fmt.get(call_node, 'indent'))
        self.assertEqual(second, fmt.get(call_node, 'indent_diff'))