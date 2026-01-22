import argparse
import io
import unittest
from unittest import mock
from cliff.formatters import commaseparated
from cliff.tests import test_columns
def test_commaseparated_list_formatter_quoted(self):
    sf = commaseparated.CSVLister()
    c = ('a', 'b', 'c')
    d1 = ('A', 'B', 'C')
    d2 = ('D', 'E', 'F')
    data = [d1, d2]
    expected = '"a","b","c"\n"A","B","C"\n"D","E","F"\n'
    output = io.StringIO()
    parser = argparse.ArgumentParser(description='Testing...')
    sf.add_argument_group(parser)
    parsed_args = parser.parse_args(['--quote', 'all'])
    sf.emit_list(c, data, output, parsed_args)
    actual = output.getvalue()
    self.assertEqual(expected, actual)