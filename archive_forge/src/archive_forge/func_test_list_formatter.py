from io import StringIO
from cliff.formatters import value
from cliff.tests import base
from cliff.tests import test_columns
def test_list_formatter(self):
    sf = value.ValueFormatter()
    c = ('a', 'b', 'c')
    d1 = ('A', 'B', 'C')
    d2 = ('D', 'E', 'F')
    data = [d1, d2]
    expected = 'A B C\nD E F\n'
    output = StringIO()
    sf.emit_list(c, data, output, None)
    actual = output.getvalue()
    self.assertEqual(expected, actual)