from oslo_log import log as logging
import six
from os_win._i18n import _
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
from os_win.utils import hostutils
from os_win.utils import pathutils
from os_win.utils import win32utils
def get_target_information(self, target_name):
    wt_host = self._get_wt_host(target_name)
    info = {}
    info['target_iqn'] = wt_host.TargetIQN
    info['enabled'] = wt_host.Enabled
    info['connected'] = bool(wt_host.Status)
    if wt_host.EnableCHAP:
        info['auth_method'] = 'CHAP'
        info['auth_username'] = wt_host.CHAPUserName
        info['auth_password'] = wt_host.CHAPSecret
    return info