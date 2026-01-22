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
def iscsi_target_exists(self, target_name):
    wt_host = self._get_wt_host(target_name, fail_if_not_found=False)
    return wt_host is not None