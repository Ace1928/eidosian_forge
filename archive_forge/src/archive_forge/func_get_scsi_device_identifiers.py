import contextlib
import ctypes
from oslo_log import log as logging
import six
from os_win._i18n import _
from os_win import _utils
import os_win.conf
from os_win import exceptions
from os_win.utils.storage import diskutils
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import libs as w_lib
from os_win.utils.winapi.libs import hbaapi as fc_struct
def get_scsi_device_identifiers(self, node_wwn, port_wwn, remote_port_wwn, fcp_lun, select_supported_identifiers=True):
    node_wwn_struct = self._wwn_struct_from_hex_str(node_wwn)
    port_wwn_struct = self._wwn_struct_from_hex_str(port_wwn)
    remote_port_wwn_struct = self._wwn_struct_from_hex_str(remote_port_wwn)
    with self._get_hba_handle(adapter_wwn_struct=node_wwn_struct) as hba_handle:
        vpd_data = self._get_scsi_device_id_vpd(hba_handle, port_wwn_struct, remote_port_wwn_struct, fcp_lun)
        identifiers = self._diskutils._parse_scsi_page_83(vpd_data, select_supported_identifiers=select_supported_identifiers)
        return identifiers