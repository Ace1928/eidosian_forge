from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
from ansible.module_utils._text import to_native
def check_service_state(self, host, service_name):
    host_service_system = host.configManager.serviceSystem
    if host_service_system:
        services = host_service_system.serviceInfo.service
        for service in services:
            if service.key == service_name:
                return (service.running, service.policy)
    msg = "Failed to find '%s' service on host system '%s'" % (service_name, host.name)
    cluster_name = self.params.get('cluster_name', None)
    if cluster_name:
        msg += " located on cluster '%s'" % cluster_name
    msg += ', please check if you have specified a valid ESXi service name.'
    self.module.fail_json(msg=msg)