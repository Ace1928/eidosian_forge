from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible.module_utils._text import to_native
def get_all_vmks_by_service_type(self):
    """
        Return information about service types and VMKernel
        Returns: Dictionary of service type as key and VMKernel list as value

        """
    service_type_vmk = dict(vmotion=[], vsan=[], management=[], faultToleranceLogging=[], vSphereProvisioning=[], vSphereReplication=[], vSphereReplicationNFC=[], vSphereBackupNFC=[])
    for service_type in list(service_type_vmk):
        vmks_list = self.query_service_type_for_vmks(service_type)
        service_type_vmk[service_type] = vmks_list
    return service_type_vmk