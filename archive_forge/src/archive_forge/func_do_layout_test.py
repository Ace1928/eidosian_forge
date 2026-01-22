from __future__ import unicode_literals
import contextlib
import logging
import unittest
import sys
from cmakelang.format import __main__
from cmakelang import configuration
from cmakelang import lex
from cmakelang import parse
from cmakelang.format import formatter
from cmakelang.parse.common import NodeType
def do_layout_test(self, input_str, expect_tree, strip_len=6):
    """
    Run the formatter on the input string and assert that the result matches
    the output string
    """
    input_str = strip_indent(input_str, strip_len)
    tokens = lex.tokenize(input_str)
    parse_tree = parse.parse(tokens, self.parse_ctx)
    box_tree = formatter.layout_tree(parse_tree, self.config)
    assert_tree(self, [box_tree], expect_tree)