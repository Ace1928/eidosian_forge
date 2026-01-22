from oslotest import base as test_base
from oslo_utils import specs_matcher
def test_specs_fails_simple3(self):
    self._do_specs_matcher_test(value='222', req='2', matches=False)