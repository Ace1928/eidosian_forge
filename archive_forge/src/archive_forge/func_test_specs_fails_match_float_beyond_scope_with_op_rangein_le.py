from oslotest import base as test_base
from oslo_utils import specs_matcher
def test_specs_fails_match_float_beyond_scope_with_op_rangein_le(self):
    self._do_specs_matcher_test(matches=False, value='20.3', req='<range-in> [ 10.1 20.2 ]')