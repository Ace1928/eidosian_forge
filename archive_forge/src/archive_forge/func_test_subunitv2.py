import os
from breezy import tests
from breezy.tests import features
from breezy.transport import memory
def test_subunitv2(self):
    self.requireFeature(features.subunit)
    params = self.get_params_passed_to_core('selftest --subunit2')
    self.assertEqual(tests.SubUnitBzrRunnerv2, params[1]['runner_class'])