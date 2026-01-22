from .. import controldir, errors, tests, ui
from .scenarios import load_tests_apply_scenarios
def test_check_support_status_supported(self):
    controldir.ControlComponentFormat().check_support_status(allow_unsupported=False)
    controldir.ControlComponentFormat().check_support_status(allow_unsupported=True)