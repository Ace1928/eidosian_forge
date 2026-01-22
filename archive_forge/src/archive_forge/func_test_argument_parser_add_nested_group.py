import unittest
from cliff import _argparse
def test_argument_parser_add_nested_group(self):
    parser = _argparse.ArgumentParser(conflict_handler='ignore')
    group = parser.add_argument_group()
    group.add_argument_group()