import os
import tempfile
from fire import __main__
from fire import testutils
def testArgPassing(self):
    expected = os.path.join('part1', 'part2', 'part3')
    with self.assertOutputMatches('%s\n' % expected):
        __main__.main(['__main__.py', 'os.path', 'join', 'part1', 'part2', 'part3'])
    with self.assertOutputMatches('%s\n' % expected):
        __main__.main(['__main__.py', 'os', 'path', '-', 'join', 'part1', 'part2', 'part3'])