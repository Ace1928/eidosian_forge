import socket
from oslo_log import log as logging
from os_win._i18n import _
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
from os_win.utils.winapi import libs as w_lib
def is_host_guarded(self):
    """Checks if the host is guarded.

        :returns: False, only Windows / Hyper-V Server 2016 or newer can be
            guarded.
        """
    return False