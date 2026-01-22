from oslotest import base as test_base
from oslo_utils import specs_matcher
def test_specs_matches_int_lower_int_range_with_op_rangein_ge(self):
    self._do_specs_matcher_test(matches=True, value='10', req='<range-in> [ 10 20 ]')