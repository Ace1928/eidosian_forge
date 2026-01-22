import os
import socket
from oslo_log import log as logging
from os_win._i18n import _
from os_win import _utils
from os_win import exceptions
from os_win.utils import baseutils
from os_win.utils import win32utils
def mount_smb_share(self, share_path, username=None, password=None):
    try:
        LOG.debug('Mounting share: %s', share_path)
        self._smb_conn.Msft_SmbMapping.Create(RemotePath=share_path, UserName=username, Password=password)
    except exceptions.x_wmi as exc:
        err_msg = _('Unable to mount SMBFS share: %(share_path)s WMI exception: %(wmi_exc)s') % {'share_path': share_path, 'wmi_exc': exc}
        raise exceptions.SMBException(err_msg)