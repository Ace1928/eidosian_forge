from oslotest import base as test_base
from oslo_utils import specs_matcher
def test_specs_matches_with_op_g3(self):
    self._do_specs_matcher_test(value='3.0', req='> 2', matches=True)