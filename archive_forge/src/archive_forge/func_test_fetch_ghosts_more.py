from .. import TestCaseWithTransport
def test_fetch_ghosts_more(self):
    self.run_bzr('init')
    with open('myfile', 'wb') as f:
        f.write(b'hello')
    self.run_bzr('add')
    self.run_bzr('commit -m hello')
    self.run_bzr('branch . my_branch')
    self.run_bzr('fetch-ghosts my_branch')