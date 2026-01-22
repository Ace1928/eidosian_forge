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
def remove_old_snapshosts(module, vm_service, snapshots_service):
    deleted_snapshots = []
    changed = False
    date_now = datetime.now()
    for snapshot in snapshots_service.list():
        if snapshot.vm is not None and snapshot.vm.name == module.params.get('vm_name'):
            diff = date_now - snapshot.date.replace(tzinfo=None)
            if diff.days >= module.params.get('keep_days_old'):
                snapshot = remove_snapshot(module, vm_service, snapshots_service, snapshot.id).get('snapshot')
                deleted_snapshots.append(snapshot)
                changed = True
    return dict(snapshots=deleted_snapshots, changed=changed)