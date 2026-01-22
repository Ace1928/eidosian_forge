import os
import socket
from oslo_log import log as logging
from os_win._i18n import _
from os_win import _utils
from os_win import exceptions
from os_win.utils import baseutils
from os_win.utils import win32utils
def unmount_smb_share(self, share_path, force=False):
    mappings = self._smb_conn.Msft_SmbMapping(RemotePath=share_path)
    if not mappings:
        LOG.debug('Share %s is not mounted. Skipping unmount.', share_path)
    for mapping in mappings:
        try:
            mapping.Remove(Force=force)
        except AttributeError:
            pass
        except exceptions.x_wmi:
            if force:
                raise exceptions.SMBException(_('Could not unmount share: %s') % share_path)