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
def test_indent_extra_newlines_with_comment(self):
    src = textwrap.dedent('        if a:\n            #not here\n\n          b\n        ')
    t = pasta.parse(src)
    if_node = t.body[0]
    b = if_node.body[0]
    self.assertEqual('  ', fmt.get(b, 'indent_diff'))