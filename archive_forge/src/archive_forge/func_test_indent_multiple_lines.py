from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import formatting
from fire import testutils
def test_indent_multiple_lines(self):
    text = formatting.Indent('hello\nworld', spaces=2)
    self.assertEqual('  hello\n  world', text)