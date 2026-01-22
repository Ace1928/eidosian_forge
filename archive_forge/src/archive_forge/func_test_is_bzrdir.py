from .. import controldir, errors, tests, ui
from .scenarios import load_tests_apply_scenarios
def test_is_bzrdir(self):
    self.assertTrue(controldir.is_control_filename('.bzr'))
    self.assertTrue(controldir.is_control_filename('.git'))