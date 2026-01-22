import socket
from oslo_log import log as logging
from os_win._i18n import _
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
from os_win.utils.winapi import libs as w_lib
def supports_nested_virtualization(self):
    """Checks if the host supports nested virtualization.

        :returns: False, only Windows / Hyper-V Server 2016 or newer supports
            nested virtualization.
        """
    return False