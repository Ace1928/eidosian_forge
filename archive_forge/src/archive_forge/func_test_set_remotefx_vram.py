from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
def test_set_remotefx_vram(self):
    self._vmutils._set_remotefx_vram(mock.sentinel.remotefx_ctrl_res, mock.sentinel.vram_bytes)