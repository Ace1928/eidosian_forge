from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils import six
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def needs_sync(nics_service):
    nics = nics_service.list()
    for nic in nics:
        nic_service = nics_service.nic_service(nic.id)
        for network_attachment_service in nic_service.network_attachments_service().list():
            if not network_attachment_service.in_sync:
                return True
    return False