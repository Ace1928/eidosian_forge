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
def delete_iscsi_target(self, target_name):
    """Removes ISCSI target."""
    try:
        wt_host = self._get_wt_host(target_name, fail_if_not_found=False)
        if not wt_host:
            LOG.debug('Skipping deleting target %s as it does not exist.', target_name)
            return
        wt_host.RemoveAllWTDisks()
        wt_host.Delete_()
    except exceptions.x_wmi as wmi_exc:
        err_msg = _('Failed to delete ISCSI target %s')
        raise exceptions.ISCSITargetWMIException(err_msg % target_name, wmi_exc=wmi_exc)