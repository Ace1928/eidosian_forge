from __future__ import unicode_literals
import unittest
from cmakelang import configuration
from cmakelang import lex
from cmakelang import parse
from cmakelang.parse.printer import tree_string, test_string
from cmakelang.parse.common import NodeType

    Run the parser to get the fst, then compare the result to the types in the
    ``expect_tree`` tuple tree.
    