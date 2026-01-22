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
def test_scope_trailing_comma(self):
    template = 'def foo(a, b{trailing_comma}): pass'
    for trailing_comma in ('', ',', ' , '):
        tree = pasta.parse(template.format(trailing_comma=trailing_comma))
        self.assertEqual(trailing_comma.lstrip(' ') + ')', fmt.get(tree.body[0], 'args_suffix'))
    template = 'class Foo(a, b{trailing_comma}): pass'
    for trailing_comma in ('', ',', ' , '):
        tree = pasta.parse(template.format(trailing_comma=trailing_comma))
        self.assertEqual(trailing_comma.lstrip(' ') + ')', fmt.get(tree.body[0], 'bases_suffix'))
    template = 'from mod import (a, b{trailing_comma})'
    for trailing_comma in ('', ',', ' , '):
        tree = pasta.parse(template.format(trailing_comma=trailing_comma))
        self.assertEqual(trailing_comma + ')', fmt.get(tree.body[0], 'names_suffix'))