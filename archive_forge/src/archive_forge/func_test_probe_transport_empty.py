from .. import controldir, errors, tests, ui
from .scenarios import load_tests_apply_scenarios
def test_probe_transport_empty(self):
    transport = self.get_transport('.')
    self.assertRaises(errors.NotBranchError, self.prober.probe_transport, transport)