from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
@mock.patch.object(_wqlutils, 'get_element_associated_class')
def test_disable_remotefx_video_adapter_not_found(self, mock_get_element_associated_class):
    mock_vm = self._lookup_vm()
    mock_get_element_associated_class.return_value = []
    self._vmutils.disable_remotefx_video_adapter(mock.sentinel.fake_vm_name)
    mock_get_element_associated_class.assert_called_once_with(self._vmutils._conn, self._vmutils._CIM_RES_ALLOC_SETTING_DATA_CLASS, element_instance_id=mock_vm.InstanceID)
    self.assertFalse(self._vmutils._jobutils.remove_virt_resource.called)