import socket
from oslo_log import log as logging
from os_win._i18n import _
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
from os_win.utils.winapi import libs as w_lib
def host_power_action(self, action):
    win32_os = self._conn_cimv2.Win32_OperatingSystem()[0]
    if action == constants.HOST_POWER_ACTION_SHUTDOWN:
        win32_os.Win32Shutdown(self._HOST_FORCED_SHUTDOWN)
    elif action == constants.HOST_POWER_ACTION_REBOOT:
        win32_os.Win32Shutdown(self._HOST_FORCED_REBOOT)
    else:
        raise NotImplementedError(_('Host %(action)s is not supported by the Hyper-V driver') % {'action': action})