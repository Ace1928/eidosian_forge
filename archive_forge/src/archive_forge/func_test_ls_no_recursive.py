from breezy import ignores, tests
def test_ls_no_recursive(self):
    self.build_tree(['subdir/', 'subdir/b'])
    self.wt.add(['a', 'subdir/', 'subdir/b', '.bzrignore'])
    self.ls_equals('.bzrignore\na\nsubdir/\n', recursive=False)
    self.ls_equals('V        .bzrignore\nV        a\nV        subdir/\n', '--verbose', recursive=False)
    self.ls_equals('b\n', working_dir='subdir')
    self.ls_equals('b\x00', '--null', working_dir='subdir')
    self.ls_equals('subdir/b\n', '--from-root', working_dir='subdir')
    self.ls_equals('subdir/b\x00', '--from-root --null', working_dir='subdir')
    self.ls_equals('subdir/b\n', '--from-root', recursive=False, working_dir='subdir')