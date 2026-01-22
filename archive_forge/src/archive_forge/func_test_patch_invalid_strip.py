from breezy.tests import TestCaseWithTransport
def test_patch_invalid_strip(self):
    self.run_bzr_error(args='patch --strip=a', error_regexes=['brz: ERROR: invalid value for option -p/--strip: a'])