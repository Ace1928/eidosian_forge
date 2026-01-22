from oslotest import base as test_base
from oslo_utils import specs_matcher
def test_specs_errors_bad_literal_with_op_allin(self):
    self.assertRaises(TypeError, specs_matcher.match, value='^&*($', req='<all-in> aes')