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
def create_security_rules(self, switch_port_name, sg_rules):
    port = self._get_switch_port_allocation(switch_port_name)[0]
    self._bind_security_rules(port, sg_rules)