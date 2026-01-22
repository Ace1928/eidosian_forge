import unittest
from cliff import _argparse
def test_argument_parser_add_mx_nested_mutually_exclusive_group(self):
    parser = _argparse.ArgumentParser(conflict_handler='ignore')
    group = parser.add_mutually_exclusive_group()
    group.add_mutually_exclusive_group()