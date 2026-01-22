import os
from breezy.tests import TestCaseWithTransport
from breezy.workingtree import WorkingTree
def test_remerge(self):
    """Remerge command works as expected"""
    self.create_conflicts()
    self.run_bzr('merge ../other --show-base', retcode=1, working_dir='this')
    with open('this/hello') as f:
        conflict_text = f.read()
    self.assertTrue('|||||||' in conflict_text)
    self.assertTrue('hi world' in conflict_text)
    self.run_bzr_error(['conflicts encountered'], 'remerge', retcode=1, working_dir='this')
    with open('this/hello') as f:
        conflict_text = f.read()
    self.assertFalse('|||||||' in conflict_text)
    self.assertFalse('hi world' in conflict_text)
    os.unlink('this/hello.OTHER')
    os.unlink('this/question.OTHER')
    self.run_bzr_error(['jello is not versioned'], 'remerge jello --merge-type weave', working_dir='this')
    self.run_bzr_error(['conflicts encountered'], 'remerge hello --merge-type weave', retcode=1, working_dir='this')
    self.assertPathExists('this/hello.OTHER')
    self.assertPathDoesNotExist('this/question.OTHER')
    file_id = self.run_bzr('file-id hello', working_dir='this')[0]
    self.run_bzr_error(['hello.THIS is not versioned'], 'file-id hello.THIS', working_dir='this')
    self.run_bzr_error(['conflicts encountered'], 'remerge --merge-type weave', retcode=1, working_dir='this')
    self.assertPathExists('this/hello.OTHER')
    self.assertTrue('this/hello.BASE')
    with open('this/hello') as f:
        conflict_text = f.read()
    self.assertFalse('|||||||' in conflict_text)
    self.assertFalse('hi world' in conflict_text)
    self.run_bzr_error(['Showing base is not supported.*Weave'], 'remerge . --merge-type weave --show-base', working_dir='this')
    self.run_bzr_error(["Can't reprocess and show base"], 'remerge . --show-base --reprocess', working_dir='this')
    self.run_bzr_error(['conflicts encountered'], 'remerge . --merge-type weave --reprocess', retcode=1, working_dir='this')
    self.run_bzr_error(['conflicts encountered'], 'remerge hello --show-base', retcode=1, working_dir='this')
    self.run_bzr_error(['conflicts encountered'], 'remerge hello --reprocess', retcode=1, working_dir='this')
    self.run_bzr('resolve --all', working_dir='this')
    self.run_bzr('commit -m done', working_dir='this')
    self.run_bzr_error(['remerge only works after normal merges', 'Not cherrypicking or multi-merges'], 'remerge', working_dir='this')