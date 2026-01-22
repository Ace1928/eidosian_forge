from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
def test_set_secure_boot_CA_required(self):
    self.assertRaises(exceptions.HyperVException, self._vmutils._set_secure_boot, mock.MagicMock(), True)