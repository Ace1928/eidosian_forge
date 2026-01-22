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
def test_simple_statement(self):
    self.do_layout_test('      cmake_minimum_required(VERSION 2.8.11)\n      ', [(NodeType.BODY, 0, 0, 0, 38, [(NodeType.STATEMENT, 0, 0, 0, 38, [(NodeType.FUNNAME, 0, 0, 0, 22, []), (NodeType.LPAREN, 0, 0, 22, 23, []), (NodeType.ARGGROUP, 0, 0, 23, 37, [(NodeType.KWARGGROUP, 0, 0, 23, 37, [(NodeType.KEYWORD, 0, 0, 23, 30, []), (NodeType.ARGGROUP, 0, 0, 31, 37, [(NodeType.PARGGROUP, 0, 0, 31, 37, [(NodeType.ARGUMENT, 0, 0, 31, 37, [])])])])]), (NodeType.RPAREN, 0, 0, 37, 38, [])])])])