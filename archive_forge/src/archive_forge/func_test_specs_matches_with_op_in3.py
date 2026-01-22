from oslotest import base as test_base
from oslo_utils import specs_matcher
def test_specs_matches_with_op_in3(self):
    self._do_specs_matcher_test(value='12311321', req='<in> 12311321 <in>', matches=True)