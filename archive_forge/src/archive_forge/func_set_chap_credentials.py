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
def set_chap_credentials(self, target_name, chap_username, chap_password):
    try:
        wt_host = self._get_wt_host(target_name)
        wt_host.EnableCHAP = True
        wt_host.CHAPUserName = chap_username
        wt_host.CHAPSecret = chap_password
        wt_host.put()
    except exceptions.x_wmi as wmi_exc:
        err_msg = _('Failed to set CHAP credentials on target %s.')
        raise exceptions.ISCSITargetWMIException(err_msg % target_name, wmi_exc=wmi_exc)