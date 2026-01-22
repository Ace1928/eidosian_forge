import os
from breezy import tests
from breezy.tests import features
from breezy.transport import memory
def test_lsprof_tests(self):
    params = self.get_params_passed_to_core('selftest --lsprof-tests')
    self.assertEqual(True, params[1]['lsprof_tests'])