from oslotest import base as test_base
from oslo_utils import specs_matcher
def test_specs_matches_int_with_int_range_with_op_rangein(self):
    self._do_specs_matcher_test(matches=True, value='15', req='<range-in> [ 10 20 ]')