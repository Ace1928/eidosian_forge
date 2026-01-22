import socket
from oslo_log import log as logging
from os_win._i18n import _
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
from os_win.utils.winapi import libs as w_lib
def get_nic_sriov_vfs(self):
    """Get host's NIC SR-IOV VFs.

        This method will ignore the vSwitches which do not have SR-IOV enabled,
        or which are poorly configured (the NIC does not support SR-IOV).

        :returns: a list of dictionaries, containing the following fields:
            - 'vswitch_name': the vSwtch name.
            - 'total_vfs': the vSwitch's maximum number of VFs. (> 0)
            - 'used_vfs': the vSwitch's number of used VFs. (<= 'total_vfs')
        """
    vfs = []
    vswitch_sds = self._conn.Msvm_VirtualEthernetSwitchSettingData(IOVPreferred=True)
    for vswitch_sd in vswitch_sds:
        hw_offload = self._conn.Msvm_EthernetSwitchHardwareOffloadData(SystemName=vswitch_sd.VirtualSystemIdentifier)[0]
        if not hw_offload.IovVfCapacity:
            LOG.warning('VSwitch %s has SR-IOV enabled, but it is not supported by the NIC or by the OS.', vswitch_sd.ElementName)
            continue
        nic_name = self._netutils.get_vswitch_external_network_name(vswitch_sd.ElementName)
        if not nic_name:
            LOG.warning('VSwitch %s is not external.', vswitch_sd.ElementName)
            continue
        nic = self._conn_scimv2.MSFT_NetAdapter(InterfaceDescription=nic_name)[0]
        vfs.append({'vswitch_name': vswitch_sd.ElementName, 'device_id': nic.PnPDeviceID, 'total_vfs': hw_offload.IovVfCapacity, 'used_vfs': hw_offload.IovVfUsage})
    return vfs