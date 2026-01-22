from oslotest import base as test_base
from oslo_utils import specs_matcher
def test_specs_matches_with_op_l3(self):
    self._do_specs_matcher_test(value='1.0', req='< 6', matches=True)