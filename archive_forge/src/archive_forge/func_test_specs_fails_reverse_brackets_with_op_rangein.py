from oslotest import base as test_base
from oslo_utils import specs_matcher
def test_specs_fails_reverse_brackets_with_op_rangein(self):
    self.assertRaises(TypeError, specs_matcher.match, value='23', req='<range-in> ) 10 20 (')