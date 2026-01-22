from __future__ import annotations
import glob
import os
from typing import Iterable
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from os_brick.initiator import linuxscsi
def get_fc_hbas_info(self):
    """Get Fibre Channel WWNs and device paths from the system, if any."""
    hbas = self.get_fc_hbas()
    hbas_info = []
    for hba in hbas:
        if hba['port_state'] == 'Online':
            wwpn = hba['port_name'].replace('0x', '')
            wwnn = hba['node_name'].replace('0x', '')
            device_path = hba['ClassDevicepath']
            device = hba['ClassDevice']
            hbas_info.append({'port_name': wwpn, 'node_name': wwnn, 'host_device': device, 'device_path': device_path})
    return hbas_info