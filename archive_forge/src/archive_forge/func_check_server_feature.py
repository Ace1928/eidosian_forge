import socket
from oslo_log import log as logging
from os_win._i18n import _
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
from os_win.utils.winapi import libs as w_lib
def check_server_feature(self, feature_id):
    """Checks if the given feature exists on the host."""
    return len(self._conn_cimv2.Win32_ServerFeature(ID=feature_id)) > 0