import collections
import time
from os_win import exceptions as os_win_exc
from os_win import utilsfactory
from oslo_log import log as logging
from os_brick import exception
from os_brick.i18n import _
from os_brick.initiator.windows import base as win_conn_base
from os_brick import utils
@utils.trace
def get_volume_paths(self, connection_properties):
    for attempt_num in range(self.device_scan_attempts):
        disk_paths = set()
        if attempt_num:
            time.sleep(self.device_scan_interval)
        self._diskutils.rescan_disks()
        volume_mappings = self._get_fc_volume_mappings(connection_properties)
        LOG.debug('Retrieved volume mappings %(vol_mappings)s for volume %(conn_props)s', dict(vol_mappings=volume_mappings, conn_props=connection_properties))
        for mapping in volume_mappings:
            device_name = mapping['device_name']
            if device_name:
                disk_paths.add(device_name)
        if not disk_paths and volume_mappings:
            fcp_lun = volume_mappings[0]['fcp_lun']
            try:
                disk_paths = self._get_disk_paths_by_scsi_id(connection_properties, fcp_lun)
                disk_paths = set(disk_paths or [])
            except os_win_exc.OSWinException as ex:
                LOG.debug('Failed to retrieve disk paths by SCSI ID. Exception: %s', ex)
        if not disk_paths:
            LOG.debug('No disk path retrieved yet.')
            continue
        if len(disk_paths) > 1:
            LOG.debug('Multiple disk paths retrieved: %s This may happen if MPIO did not claim them yet.', disk_paths)
            continue
        dev_num = self._diskutils.get_device_number_from_device_name(list(disk_paths)[0])
        if self.use_multipath and (not self._diskutils.is_mpio_disk(dev_num)):
            LOG.debug('Multipath was requested but the disk %s was not claimed yet by the MPIO service.', dev_num)
            continue
        return list(disk_paths)
    return []