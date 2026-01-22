import argparse
import enum
import os
import os.path
import pickle
import re
import sys
import types
import pyomo.common.unittest as unittest
from io import StringIO
from pyomo.common.dependencies import yaml, yaml_available, yaml_load_args
from pyomo.common.config import (
from pyomo.common.log import LoggingIntercept
def test_argparse_lists(self):
    c = ConfigDict()
    self.assertEqual(c.domain_name(), '')
    sub_dict = c.declare('sub_dict', ConfigDict())
    sub_dict.declare('a', ConfigValue(domain=int))
    sub_dict.declare('b', ConfigValue())
    self.assertEqual(c.sub_dict.domain_name(), 'sub-dict')
    self.assertEqual(c.sub_dict.get('a').domain_name(), 'int')
    self.assertEqual(c.sub_dict.get('b').domain_name(), '')
    c.declare('lst', ConfigList(domain=int)).declare_as_argument(action='append')
    c.declare('sub', ConfigList(domain=c.sub_dict)).declare_as_argument(action='append')
    c.declare('listof', ConfigValue(domain=ListOf(int))).declare_as_argument()
    parser = argparse.ArgumentParser(prog='tester')
    c.initialize_argparse(parser)
    self.assertIn('\n  -h, --help            show this help message and exit\n  --lst INT\n  --sub SUB-DICT\n  --listof LISTOF[INT]'.strip(), parser.format_help())
    args = parser.parse_args(['--lst', '42', '--lst', '1', '--sub', 'a=4', '--sub', 'b=12,a:0', '--listof', '3,2 4'])
    leftovers = c.import_argparse(args)
    self.assertEqual(c.lst.value(), [42, 1])
    self.assertEqual(c.sub.value(), [{'a': 4, 'b': None}, {'a': 0, 'b': '12'}])
    self.assertEqual(c.listof, [3, 2, 4])
    args = parser.parse_args(['--sub', 'b=12,a 0'])
    with self.assertRaisesRegex(ValueError, "(?s)invalid value for configuration 'sub':.*Expected ':' or '=' but found '0' at Line 1 Column 8"):
        leftovers = c.import_argparse(args)
    args = parser.parse_args(['--sub', 'b='])
    with self.assertRaisesRegex(ValueError, "(?s)Expected value following '=' but encountered end of string"):
        leftovers = c.import_argparse(args)
    args = parser.parse_args(['--sub', 'b'])
    with self.assertRaisesRegex(ValueError, "(?s)Expected ':' or '=' but encountered end of string"):
        leftovers = c.import_argparse(args)