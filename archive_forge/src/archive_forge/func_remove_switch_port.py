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
def remove_switch_port(self, switch_port_name, vnic_deleted=False):
    """Removes the switch port."""
    sw_port, found = self._get_switch_port_allocation(switch_port_name, expected=False)
    if not sw_port:
        return
    if not vnic_deleted:
        try:
            self._jobutils.remove_virt_resource(sw_port)
        except exceptions.x_wmi:
            pass
    self._switch_ports.pop(switch_port_name, None)
    self._profile_sds.pop(sw_port.InstanceID, None)
    self._vlan_sds.pop(sw_port.InstanceID, None)
    self._vsid_sds.pop(sw_port.InstanceID, None)
    self._bandwidth_sds.pop(sw_port.InstanceID, None)
    self._hw_offload_sds.pop(sw_port.InstanceID, None)