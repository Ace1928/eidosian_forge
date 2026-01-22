from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def construct_host_interfaces(self, interfaces):
    """Ensures interfaces object is properly formatted before submitting it to API.

        Args:
            interfaces (list): list of dictionaries for each interface present on the host.

        Returns:
            (interfaces, ip) - where interfaces is original list reformated into a valid format
                and ip is any IP address found on interface of type agent (printing purposes only).
        """
    ip = ''
    interface_types = {'agent': 1, 'snmp': 2, 'ipmi': 3, 'jmx': 4}
    type_to_port = {1: '10050', 2: '161', 3: '623', 4: '12345'}
    for interface in interfaces:
        if interface['type'] in list(interface_types.keys()):
            interface['type'] = interface_types[interface['type']]
        else:
            interface['type'] = int(interface['type'])
        if interface['type'] == 1:
            ip = interface.get('ip', '')
        for key in ['ip', 'dns']:
            if key not in interface or interface[key] is None:
                interface[key] = ''
        if 'port' not in interface or interface['port'] is None:
            interface['port'] = type_to_port.get(interface['type'], '')
        if 'bulk' in interface:
            del interface['bulk']
        if interface['type'] == 2:
            if not interface['details']:
                self._module.fail_json(msg="Option 'details' required for SNMP interface {0}".format(interface))
            i_details = interface['details']
            if i_details['version'] < 3 and (not i_details.get('community', False)):
                self._module.fail_json(msg="Option 'community' is required in 'details' for SNMP interface {0}".format(interface))
        else:
            interface['details'] = {}
    return (interfaces, ip)