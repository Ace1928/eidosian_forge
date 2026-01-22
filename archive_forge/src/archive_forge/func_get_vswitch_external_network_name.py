import functools
import re
from eventlet import patcher
from eventlet import tpool
from oslo_log import log as logging
from oslo_utils import units
import six
from os_win._i18n import _
from os_win import conf
from os_win import constants
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils import baseutils
from os_win.utils import jobutils
def get_vswitch_external_network_name(self, vswitch_name):
    ext_port = self._get_vswitch_external_port(vswitch_name)
    if ext_port:
        return ext_port.ElementName