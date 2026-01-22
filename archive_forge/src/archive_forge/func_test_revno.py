import os
from breezy import tests
from breezy.bzr.tests.matchers import ContainsNoVfsCalls
from breezy.errors import NoSuchRevision
def test_revno(self):

    def bzr(*args, **kwargs):
        return self.run_bzr(*args, **kwargs)[0]
    os.mkdir('a')
    os.chdir('a')
    bzr('init')
    self.assertEqual(int(bzr('revno')), 0)
    with open('foo', 'wb') as f:
        f.write(b'foo\n')
    bzr('add foo')
    bzr('commit -m foo')
    self.assertEqual(int(bzr('revno')), 1)
    os.mkdir('baz')
    bzr('add baz')
    bzr('commit -m baz')
    self.assertEqual(int(bzr('revno')), 2)
    os.chdir('..')
    self.assertEqual(int(bzr('revno a')), 2)
    self.assertEqual(int(bzr('revno a/baz')), 2)