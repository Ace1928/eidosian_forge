from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
def test_get_free_controller_slot_exception(self):
    fake_drive = mock.MagicMock()
    type(fake_drive).AddressOnParent = mock.PropertyMock(side_effect=list(range(constants.SCSI_CONTROLLER_SLOTS_NUMBER)))
    with mock.patch.object(self._vmutils, 'get_attached_disks') as fake_get_attached_disks:
        fake_get_attached_disks.return_value = [fake_drive] * constants.SCSI_CONTROLLER_SLOTS_NUMBER
        self.assertRaises(exceptions.HyperVException, self._vmutils.get_free_controller_slot, mock.sentinel.scsi_controller_path)