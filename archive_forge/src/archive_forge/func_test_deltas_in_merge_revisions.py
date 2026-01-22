import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def test_deltas_in_merge_revisions(self):
    """Check deltas created for both mainline and merge revisions"""
    wt = self.make_branch_and_tree('parent')
    self.build_tree(['parent/file1', 'parent/file2', 'parent/file3'])
    wt.add('file1')
    wt.add('file2')
    wt.commit(message='add file1 and file2')
    self.run_bzr('branch parent child')
    os.unlink('child/file1')
    with open('child/file2', 'wb') as f:
        f.write(b'hello\n')
    self.run_bzr(['commit', '-m', 'remove file1 and modify file2', 'child'])
    os.chdir('parent')
    self.run_bzr('merge ../child')
    wt.commit('merge child branch')
    os.chdir('..')
    b = wt.branch
    lf = LogCatcher()
    lf.supports_merge_revisions = True
    log.show_log(b, lf, verbose=True)
    revs = lf.revisions
    self.assertEqual(3, len(revs))
    logentry = revs[0]
    self.assertEqual('2', logentry.revno)
    self.assertEqual('merge child branch', logentry.rev.message)
    self.checkDelta(logentry.delta, removed=['file1'], modified=['file2'])
    logentry = revs[1]
    self.assertEqual('1.1.1', logentry.revno)
    self.assertEqual('remove file1 and modify file2', logentry.rev.message)
    self.checkDelta(logentry.delta, removed=['file1'], modified=['file2'])
    logentry = revs[2]
    self.assertEqual('1', logentry.revno)
    self.assertEqual('add file1 and file2', logentry.rev.message)
    self.checkDelta(logentry.delta, added=['file1', 'file2'])