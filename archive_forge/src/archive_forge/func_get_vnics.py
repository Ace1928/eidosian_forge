from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def get_vnics(networks_service, network, connection):
    resp = []
    vnic_services = connection.system_service().vnic_profiles_service()
    for vnic in vnic_services.list():
        if vnic.network.id == network.id:
            resp.append(vnic)
    return resp