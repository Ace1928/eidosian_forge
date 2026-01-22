from oslotest import base as test_base
from oslo_utils import specs_matcher
def test_specs_matches_with_op_sg(self):
    self._do_specs_matcher_test(value='2', req='s> 12', matches=True)