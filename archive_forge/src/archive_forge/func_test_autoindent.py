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
def test_autoindent(self):
    src = textwrap.dedent('        def a():\n            b\n            c\n        ')
    expected = textwrap.dedent('        def a():\n            b\n            new_node\n        ')
    t = pasta.parse(src)
    t.body[0].body[1] = ast.Expr(ast.Name(id='new_node'))
    self.assertMultiLineEqual(expected, codegen.to_str(t))