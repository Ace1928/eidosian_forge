from unittest import mock
import ddt
import six
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils10
from os_win.utils import jobutils
def test_validate_remotefx_vram(self):
    self.assertRaises(exceptions.HyperVRemoteFXException, self._vmutils._validate_remotefx_params, 1, constants.REMOTEFX_MAX_RES_1024x768, vram_bytes=10000)