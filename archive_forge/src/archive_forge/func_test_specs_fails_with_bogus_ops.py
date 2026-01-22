from oslotest import base as test_base
from oslo_utils import specs_matcher
def test_specs_fails_with_bogus_ops(self):
    self._do_specs_matcher_test(value='4', req='! 2', matches=False)