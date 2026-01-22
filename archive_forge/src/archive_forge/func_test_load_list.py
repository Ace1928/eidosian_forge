import os
from breezy import tests
from breezy.tests import features
from breezy.transport import memory
def test_load_list(self):
    params = self.get_params_passed_to_core('selftest --load-list foo')
    self.assertEqual('foo', params[1]['load_list'])