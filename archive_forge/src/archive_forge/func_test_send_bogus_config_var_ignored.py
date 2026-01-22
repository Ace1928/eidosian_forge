from io import BytesIO
from ... import branch, merge_directive, tests
from ...bzr.bundle import serializer
from ...controldir import ControlDir
from ...transport import memory
from .. import scenarios
def test_send_bogus_config_var_ignored(self):
    self.set_config_send_strict("I'm unsure")
    self.assertSendSucceeds([], with_warning=True)