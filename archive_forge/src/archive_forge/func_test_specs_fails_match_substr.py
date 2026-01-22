from oslotest import base as test_base
from oslo_utils import specs_matcher
def test_specs_fails_match_substr(self):
    self._do_specs_matcher_test(value=str(['X___X']), req='<all-in> ___', matches=False)