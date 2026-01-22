from __future__ import unicode_literals
import unittest
from cmakelang import configuration
from cmakelang import lex
from cmakelang import parse
from cmakelang.parse.printer import tree_string, test_string
from cmakelang.parse.common import NodeType
def do_type_test(self, input_str, expect_tree):
    """
    Run the parser to get the fst, then compare the result to the types in the
    ``expect_tree`` tuple tree.
    """
    tokens = lex.tokenize(input_str)
    fst_root = parse.parse(tokens, self.parse_ctx)
    assert_tree_type(self, [fst_root], expect_tree)