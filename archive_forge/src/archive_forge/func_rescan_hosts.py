from __future__ import annotations
import glob
import os
from typing import Iterable
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from os_brick.initiator import linuxscsi
def rescan_hosts(self, hbas: Iterable, connection_properties: dict) -> None:
    LOG.debug('Rescanning HBAs %(hbas)s with connection properties %(conn_props)s', {'hbas': hbas, 'conn_props': connection_properties})
    ports = connection_properties.get('initiator_target_lun_map')
    if ports:
        hbas = [hba for hba in hbas if hba['port_name'] in ports]
        LOG.debug('Using initiator target map to exclude HBAs: %s', hbas)
    broad_scan = connection_properties.get('enable_wildcard_scan', True)
    if not broad_scan:
        LOG.debug('Connection info disallows broad SCSI scanning')
    process = []
    skipped = []
    get_ctls = self._get_hba_channel_scsi_target_lun
    for hba in hbas:
        ctls, luns_wildcards = get_ctls(hba, connection_properties)
        if ctls:
            process.append((hba, ctls))
        elif not broad_scan:
            LOG.debug('Skipping HBA %s, nothing to scan, target port not connected to initiator', hba['node_name'])
        elif not process:
            skipped.append((hba, [('-', '-', lun) for lun in luns_wildcards]))
    process = process or skipped
    addressing_mode = connection_properties.get('addressing_mode')
    for hba, ctls in process:
        for hba_channel, target_id, target_lun in ctls:
            target_lun = self.lun_for_addressing(target_lun, addressing_mode)
            LOG.debug('Scanning %(host)s (wwnn: %(wwnn)s, c: %(channel)s, t: %(target)s, l: %(lun)s)', {'host': hba['host_device'], 'wwnn': hba['node_name'], 'channel': hba_channel, 'target': target_id, 'lun': target_lun})
            self.echo_scsi_command('/sys/class/scsi_host/%s/scan' % hba['host_device'], '%(c)s %(t)s %(l)s' % {'c': hba_channel, 't': target_id, 'l': target_lun})