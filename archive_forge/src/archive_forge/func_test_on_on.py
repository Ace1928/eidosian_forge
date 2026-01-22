from oslotest import base as test_base
import oslo_i18n
from oslo_i18n import _gettextutils
from oslo_i18n._i18n import _
from oslo_i18n import _lazy
from oslo_i18n import _message
from oslo_i18n import _translate
from oslo_i18n import fixture
def test_on_on(self):
    _lazy.USE_LAZY = True
    f = fixture.ToggleLazy(True)
    f.setUp()
    self.assertTrue(_lazy.USE_LAZY)
    f._restore_original()
    self.assertTrue(_lazy.USE_LAZY)