import os
import breezy.osutils
from breezy.tests import TestCaseWithTransport
from breezy.trace import mutter
from breezy.workingtree import WorkingTree
def test_revert(self):
    self.run_bzr('init')
    with open('hello', 'w') as f:
        f.write('foo')
    self.run_bzr('add hello')
    self.run_bzr('commit -m setup hello')
    with open('goodbye', 'w') as f:
        f.write('baz')
    self.run_bzr('add goodbye')
    self.run_bzr('commit -m setup goodbye')
    with open('hello', 'w') as f:
        f.write('bar')
    with open('goodbye', 'w') as f:
        f.write('qux')
    self.run_bzr('revert hello')
    self.check_file_contents('hello', b'foo')
    self.check_file_contents('goodbye', b'qux')
    self.run_bzr('revert')
    self.check_file_contents('goodbye', b'baz')
    os.mkdir('revertdir')
    self.run_bzr('add revertdir')
    self.run_bzr('commit -m f')
    os.rmdir('revertdir')
    self.run_bzr('revert')
    if breezy.osutils.supports_symlinks(self.test_dir):
        os.symlink('/unlikely/to/exist', 'symlink')
        self.run_bzr('add symlink')
        self.run_bzr('commit -m f')
        os.unlink('symlink')
        self.run_bzr('revert')
        self.assertPathExists('symlink')
        os.unlink('symlink')
        os.symlink('a-different-path', 'symlink')
        self.run_bzr('revert')
        self.assertEqual('/unlikely/to/exist', os.readlink('symlink'))
    else:
        self.log('skipping revert symlink tests')
    with open('hello', 'w') as f:
        f.write('xyz')
    self.run_bzr('commit -m xyz hello')
    self.run_bzr('revert -r 1 hello')
    self.check_file_contents('hello', b'foo')
    self.run_bzr('revert hello')
    self.check_file_contents('hello', b'xyz')
    os.chdir('revertdir')
    self.run_bzr('revert')
    os.chdir('..')