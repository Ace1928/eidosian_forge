from breezy import ignores, tests
def test_ls_path_nonrecursive(self):
    self.ls_equals('%s/.bzrignore\n%s/a\n' % (self.test_dir, self.test_dir), self.test_dir, recursive=False)