from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
@mock.patch('time.sleep')
@mock.patch.object(vmutils, 'tpool')
@mock.patch.object(vmutils, 'patcher')
def test_vm_power_state_change_event_handler(self, mock_patcher, mock_tpool, mock_sleep):
    enabled_state = constants.HYPERV_VM_STATE_ENABLED
    hv_enabled_state = self._vmutils._vm_power_states_map[enabled_state]
    fake_event = mock.Mock(ElementName=mock.sentinel.vm_name, EnabledState=hv_enabled_state)
    fake_callback = mock.Mock(side_effect=Exception)
    fake_listener = self._vmutils._conn.Msvm_ComputerSystem.watch_for.return_value
    mock_tpool.execute.side_effect = (exceptions.x_wmi_timed_out, fake_event, Exception, KeyboardInterrupt)
    handler = self._vmutils.get_vm_power_state_change_listener(get_handler=True)
    self.assertRaises(KeyboardInterrupt, handler, fake_callback)
    fake_callback.assert_called_once_with(mock.sentinel.vm_name, enabled_state)
    mock_tpool.execute.assert_has_calls(fake_listener, [mock.call(constants.DEFAULT_WMI_EVENT_TIMEOUT_MS)] * 4)
    mock_sleep.assert_called_once_with(constants.DEFAULT_WMI_EVENT_TIMEOUT_MS / 1000)