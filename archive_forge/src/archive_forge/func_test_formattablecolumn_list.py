import io
import json
from unittest import mock
from cliff.formatters import json_format
from cliff.tests import base
from cliff.tests import test_columns
def test_formattablecolumn_list(self):
    sf = json_format.JSONFormatter()
    c = ('a', 'b', 'c')
    d = (('A1', 'B1', test_columns.FauxColumn(['the', 'value'])),)
    expected = [{'a': 'A1', 'b': 'B1', 'c': ['the', 'value']}]
    args = mock.Mock()
    sf.add_argument_group(args)
    args.noindent = True
    output = io.StringIO()
    sf.emit_list(c, d, output, args)
    value = output.getvalue()
    self.assertEqual(1, len(value.splitlines()))
    actual = json.loads(value)
    self.assertEqual(expected, actual)