from breezy.tests import TestCase
from breezy.textmerge import Merge2
def test_conflict(self):
    lines_a = 'a\nb\nc\nd\ne\nf\ng\nh\n'.splitlines(True)
    lines_b = 'z\nb\nx\nd\ne\ne\nf\ng\ny\n'.splitlines(True)
    expected = '<\na\n=\nz\n>\nb\n<\nc\n=\nx\n>\nd\ne\n<\n=\ne\n>\nf\ng\n<\nh\n=\ny\n>\n'
    m2 = Merge2(lines_a, lines_b, '<\n', '>\n', '=\n')
    mlines = m2.merge_lines()[0]
    self.assertEqualDiff(''.join(mlines), expected)
    mlines = m2.merge_lines(reprocess=True)[0]
    self.assertEqualDiff(''.join(mlines), expected)