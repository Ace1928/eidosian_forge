from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import formatting
from fire import testutils
def test_ellipsis_middle_truncate(self):
    text = '1000000000L'
    truncated_text = formatting.EllipsisMiddleTruncate(text=text, available_space=7, line_length=LINE_LENGTH)
    self.assertEqual('10...0L', truncated_text)