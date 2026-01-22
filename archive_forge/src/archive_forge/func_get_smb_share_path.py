import os
import socket
from oslo_log import log as logging
from os_win._i18n import _
from os_win import _utils
from os_win import exceptions
from os_win.utils import baseutils
from os_win.utils import win32utils
def get_smb_share_path(self, share_name):
    shares = self._smb_conn.Msft_SmbShare(Name=share_name)
    share_path = shares[0].Path if shares else None
    if not shares:
        LOG.debug('Could not find any local share named %s.', share_name)
    return share_path