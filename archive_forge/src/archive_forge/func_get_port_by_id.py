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
def get_port_by_id(self, port_id, vswitch_name):
    vswitch = self._get_vswitch(vswitch_name)
    switch_ports = self._conn.Msvm_EthernetSwitchPort(SystemName=vswitch.Name)
    for switch_port in switch_ports:
        if switch_port.ElementName == port_id:
            return switch_port