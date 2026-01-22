from __future__ import (absolute_import, division, print_function)
import traceback
import os
import ssl
import time
from ansible.module_utils.six.moves.http_client import HTTPSConnection, IncompleteRead
from ansible.module_utils.six.moves.urllib.parse import urlparse
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def get_snapshot_disk_id(module, snapshots_service):
    snapshot_service = snapshots_service.snapshot_service(module.params.get('snapshot_id'))
    snapshot_disks_service = snapshot_service.disks_service()
    disk_id = ''
    if module.params.get('disk_id'):
        disk_id = module.params.get('disk_id')
    elif module.params.get('disk_name'):
        disk_id = get_id_by_name(snapshot_disks_service, module.params.get('disk_name'))
    return disk_id