from .. import controldir, errors, tests, ui
from .scenarios import load_tests_apply_scenarios
def test_check_support_status_unsupported(self):
    self.assertRaises(errors.UnsupportedFormatError, UnsupportedControlComponentFormat().check_support_status, allow_unsupported=False)
    UnsupportedControlComponentFormat().check_support_status(allow_unsupported=True)