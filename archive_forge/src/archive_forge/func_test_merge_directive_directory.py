import os
import smtplib
from breezy import gpg, merge_directive, tests, workingtree
def test_merge_directive_directory(self):
    """Test --directory option"""
    import re
    re_timestamp = re.compile('^# timestamp: .*', re.M)
    self.prepare_merge_directive()
    md1 = self.run_bzr('merge-directive ../tree2')[0]
    md1 = re_timestamp.sub('# timestamp: XXX', md1)
    os.chdir('..')
    md2 = self.run_bzr('merge-directive --directory tree1 tree2')[0]
    md2 = re_timestamp.sub('# timestamp: XXX', md2)
    self.assertEqualDiff(md1.replace('../tree2', 'tree2'), md2)