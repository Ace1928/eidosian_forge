from breezy import ignores, tests
def ls_equals(self, value, args=None, recursive=True, working_dir=None):
    command = 'ls'
    if args is not None:
        command += ' ' + args
    if recursive:
        command += ' -R'
    out, err = self.run_bzr(command, working_dir=working_dir)
    self.assertEqual('', err)
    self.assertEqualDiff(value, out)