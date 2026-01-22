from breezy import ignores, tests
def test_ls_null_verbose(self):
    self.run_bzr_error(['Cannot set both --verbose and --null'], 'ls --verbose --null')