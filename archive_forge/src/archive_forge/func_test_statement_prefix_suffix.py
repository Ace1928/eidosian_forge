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
def test_statement_prefix_suffix(self):
    src = 'a\n\ndef foo():\n  return bar\n\n\nb\n'
    t = pasta.parse(src)
    self.assertEqual('\n', fmt.get(t.body[1], 'prefix'))
    self.assertEqual('', fmt.get(t.body[1], 'suffix'))