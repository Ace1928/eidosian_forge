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
def init_caches(self):
    if not self._enable_cache:
        LOG.info('WMI caching is disabled.')
        return
    for vswitch in self._conn.Msvm_VirtualEthernetSwitch():
        self._switches[vswitch.ElementName] = vswitch
    for port in self._conn.Msvm_EthernetPortAllocationSettingData():
        self._switch_ports[port.ElementName] = port
    switch_port_id_regex = re.compile('Microsoft:[0-9A-F-]*\\\\[0-9A-F-]*\\\\[0-9A-F-]', flags=re.IGNORECASE)
    for profile in self._conn.Msvm_EthernetSwitchPortProfileSettingData():
        match = switch_port_id_regex.match(profile.InstanceID)
        if match:
            self._profile_sds[match.group()] = profile
    for vlan_sd in self._conn.Msvm_EthernetSwitchPortVlanSettingData():
        match = switch_port_id_regex.match(vlan_sd.InstanceID)
        if match:
            self._vlan_sds[match.group()] = vlan_sd
    for vsid_sd in self._conn.Msvm_EthernetSwitchPortSecuritySettingData():
        match = switch_port_id_regex.match(vsid_sd.InstanceID)
        if match:
            self._vsid_sds[match.group()] = vsid_sd
    bandwidths = self._conn.Msvm_EthernetSwitchPortBandwidthSettingData()
    for bandwidth_sd in bandwidths:
        match = switch_port_id_regex.match(bandwidth_sd.InstanceID)
        if match:
            self._bandwidth_sds[match.group()] = bandwidth_sd
    hw_offloads = self._conn.Msvm_EthernetSwitchPortOffloadSettingData()
    for hw_offload_sd in hw_offloads:
        match = switch_port_id_regex.match(hw_offload_sd.InstanceID)
        if match:
            self._hw_offload_sds[match.group()] = hw_offload_sd