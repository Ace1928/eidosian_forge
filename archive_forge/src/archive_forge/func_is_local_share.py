import os
import socket
from oslo_log import log as logging
from os_win._i18n import _
from os_win import _utils
from os_win import exceptions
from os_win.utils import baseutils
from os_win.utils import win32utils
def is_local_share(self, share_path):
    if share_path in self._loopback_share_map:
        return self._loopback_share_map[share_path]
    addr = share_path.lstrip('\\').split('\\', 1)[0]
    local_ips = _utils.get_ips(socket.gethostname())
    local_ips += _utils.get_ips('localhost')
    dest_ips = _utils.get_ips(addr)
    is_local = bool(set(local_ips).intersection(set(dest_ips)))
    self._loopback_share_map[share_path] = is_local
    return is_local