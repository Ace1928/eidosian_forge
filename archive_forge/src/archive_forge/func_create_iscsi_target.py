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
def create_iscsi_target(self, target_name, fail_if_exists=False):
    """Creates ISCSI target."""
    try:
        self._conn_wmi.WT_Host.NewHost(HostName=target_name)
    except exceptions.x_wmi as wmi_exc:
        err_code = _utils.get_com_error_code(wmi_exc.com_error)
        target_exists = err_code == self._ERR_FILE_EXISTS
        if not target_exists or fail_if_exists:
            err_msg = _('Failed to create iSCSI target: %s.')
            raise exceptions.ISCSITargetWMIException(err_msg % target_name, wmi_exc=wmi_exc)
        else:
            LOG.info('The iSCSI target %s already exists.', target_name)