from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
def test_list_instance_notes(self):
    vs = mock.MagicMock()
    attrs = {'ElementName': 'fake_name', 'Notes': ['4f54fb69-d3a2-45b7-bb9b-b6e6b3d893b3']}
    vs.configure_mock(**attrs)
    vs2 = mock.MagicMock(ElementName='fake_name2', Notes=None)
    self._vmutils._conn.Msvm_VirtualSystemSettingData.return_value = [vs, vs2]
    response = self._vmutils.list_instance_notes()
    self.assertEqual([(attrs['ElementName'], attrs['Notes'])], response)
    self._vmutils._conn.Msvm_VirtualSystemSettingData.assert_called_with(['ElementName', 'Notes'], VirtualSystemType=self._vmutils._VIRTUAL_SYSTEM_TYPE_REALIZED)