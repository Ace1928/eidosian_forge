from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import formatting
from fire import testutils
def test_wrap_multiple_items(self):
    lines = formatting.WrappedJoin(['rice', 'beans', 'chicken', 'cheese'], width=15)
    self.assertEqual(['rice | beans |', 'chicken |', 'cheese'], lines)