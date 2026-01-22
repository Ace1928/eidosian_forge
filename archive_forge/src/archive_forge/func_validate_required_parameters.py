from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, netapp_ipaddress
def validate_required_parameters(self, keys):
    """
            Validate if required parameters for create or modify are present.
            Parameter requirement might vary based on given data-protocol.
            :return: None
        """
    home_node = self.parameters.get('home_node') or self.get_home_node_for_cluster()
    errors = []
    if self.use_rest and home_node is None and (self.parameters.get('home_port') is not None):
        errors.append('Cannot guess home_node, home_node is required when home_port is present with REST.')
    if 'broadcast_domain_home_port_or_home_node' in keys:
        if all((x not in self.parameters for x in ['broadcast_domain', 'home_port', 'home_node'])):
            errors.append("At least one of 'broadcast_domain', 'home_port', 'home_node' is required to create an IP interface.")
        keys.remove('broadcast_domain_home_port_or_home_node')
    if not keys.issubset(set(self.parameters.keys())):
        errors.append('Missing one or more required parameters for creating interface: %s.' % ', '.join(keys))
    if 'interface_type' in keys and 'interface_type' in self.parameters:
        if self.parameters['interface_type'] not in ['fc', 'ip']:
            errors.append('unexpected value for interface_type: %s.' % self.parameters['interface_type'])
        elif self.parameters['interface_type'] == 'fc':
            if not self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 8, 0):
                if 'home_port' in self.parameters:
                    errors.append("'home_port' is not supported for FC interfaces with 9.7, use 'current_port', avoid home_node.")
                if 'home_node' in self.parameters:
                    self.module.warn("Avoid 'home_node' with FC interfaces with 9.7, use 'current_node'.")
            if 'vserver' not in self.parameters:
                errors.append("A data 'vserver' is required for FC interfaces.")
            if 'service_policy' in self.parameters:
                errors.append("'service_policy' is not supported for FC interfaces.")
            if 'role' in self.parameters and self.parameters.get('role') != 'data':
                errors.append("'role' is deprecated, and 'data' is the only value supported for FC interfaces: found %s." % self.parameters.get('role'))
            if 'probe_port' in self.parameters:
                errors.append("'probe_port' is not supported for FC interfaces.")
    if errors:
        self.module.fail_json(msg='Error: %s' % '  '.join(errors))