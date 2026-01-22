from oslotest import base as test_base
from oslo_utils import specs_matcher
def test_specs_fails_match_substr_reversed(self):
    self._do_specs_matcher_test(value=str(['aes', 'mmx', 'aux']), req='<all-in> XaesX', matches=False)