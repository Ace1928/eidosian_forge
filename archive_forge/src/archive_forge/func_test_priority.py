from .. import controldir, errors, tests, ui
from .scenarios import load_tests_apply_scenarios
def test_priority(self):
    transport = self.get_transport('.')
    self.assertIsInstance(self.prober.priority(transport), int)