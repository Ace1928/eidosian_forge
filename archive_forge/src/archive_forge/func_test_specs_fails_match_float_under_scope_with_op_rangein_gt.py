from oslotest import base as test_base
from oslo_utils import specs_matcher
def test_specs_fails_match_float_under_scope_with_op_rangein_gt(self):
    self._do_specs_matcher_test(matches=False, value='5.0', req='<range-in> ( 5.1 20.2 ]')