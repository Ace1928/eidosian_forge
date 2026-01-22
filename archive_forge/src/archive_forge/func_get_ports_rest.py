from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_ports_rest(self, ports):
    missing_ports = []
    desired_ports = []
    for port in ports:
        current = self.get_net_port_rest(port)
        if current is None:
            missing_ports.append(port)
        else:
            desired_ports.append(current)
    if missing_ports and self.parameters['state'] == 'present':
        self.module.fail_json(msg='Error: ports: %s not found' % ', '.join(missing_ports))
    return desired_ports