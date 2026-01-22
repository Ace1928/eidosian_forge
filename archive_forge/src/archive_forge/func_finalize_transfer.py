from __future__ import (absolute_import, division, print_function)
import json
import os
import subprocess
import time
import traceback
import inspect
from ansible.module_utils.six.moves.http_client import HTTPSConnection, IncompleteRead
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def finalize_transfer(connection, module, transfer_id):
    transfer = None
    transfer_service = connection.system_service().image_transfers_service().image_transfer_service(transfer_id)
    start = time.time()
    transfer_service.finalize()
    while True:
        time.sleep(1)
        try:
            transfer = transfer_service.get()
        except sdk.NotFoundError:
            disk_service = connection.system_service().disks_service().disk_service(module.params['id'])
            try:
                disk = disk_service.get()
            except sdk.NotFoundError:
                raise RuntimeError('Transfer {0} failed: disk {1} was removed'.format(transfer.id, module.params['id']))
            if disk.status == otypes.DiskStatus.OK:
                break
            raise RuntimeError("Transfer {0} failed: disk {1} is '{2}'".format(transfer.id, module.params['id'], disk.status))
        if transfer.phase == otypes.ImageTransferPhase.FINISHED_SUCCESS:
            break
        if transfer.phase == otypes.ImageTransferPhase.FINISHED_FAILURE:
            raise RuntimeError('Transfer {0} failed, phase: {1}'.format(transfer.id, transfer.phase))
        if time.time() > start + module.params.get('timeout'):
            raise RuntimeError('Timed out waiting for transfer {0} to finalize, phase: {1}'.format(transfer.id, transfer.phase))