from oslotest import base as test_base
from oslo_i18n import _lazy
def test_disable_lazy(self):
    _lazy.USE_LAZY = True
    _lazy.enable_lazy(False)
    self.assertFalse(_lazy.USE_LAZY)