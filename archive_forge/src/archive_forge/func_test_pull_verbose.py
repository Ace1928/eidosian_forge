import os
import re
import sys
import breezy
from breezy import osutils
from breezy.branch import Branch
from breezy.errors import CommandError
from breezy.tests import TestCaseWithTransport
from breezy.tests.http_utils import TestCaseWithWebserver
from breezy.tests.test_sftp_transport import TestCaseWithSFTPServer
from breezy.workingtree import WorkingTree
def test_pull_verbose(self):
    """Pull changes from one branch to another and watch the output."""
    os.mkdir('a')
    os.chdir('a')
    self.example_branch()
    os.chdir('..')
    self.run_bzr('branch a b')
    os.chdir('b')
    with open('b', 'wb') as f:
        f.write(b'else\n')
    self.run_bzr('add b')
    self.run_bzr(['commit', '-m', 'added b'])
    os.chdir('../a')
    out = self.run_bzr('pull --verbose ../b')[0]
    self.assertNotEqual(out.find('Added Revisions:'), -1)
    self.assertNotEqual(out.find('message:\n  added b'), -1)
    self.assertNotEqual(out.find('added b'), -1)
    self.run_bzr('commit -m foo --unchanged')
    os.chdir('../b')
    self.run_bzr('commit -m baz --unchanged')
    self.run_bzr('pull ../a', retcode=3)
    out = self.run_bzr('pull --overwrite --verbose ../a')[0]
    remove_loc = out.find('Removed Revisions:')
    self.assertNotEqual(remove_loc, -1)
    added_loc = out.find('Added Revisions:')
    self.assertNotEqual(added_loc, -1)
    removed_message = out.find('message:\n  baz')
    self.assertNotEqual(removed_message, -1)
    self.assertTrue(remove_loc < removed_message < added_loc)
    added_message = out.find('message:\n  foo')
    self.assertNotEqual(added_message, -1)
    self.assertTrue(added_loc < added_message)