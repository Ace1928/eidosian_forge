from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, netapp_ipaddress
def derive_interface_type(self):
    protocols = self.parameters.get('protocols')
    if protocols in (None, ['none']):
        if self.parameters.get('role') in ('cluster', 'intercluster') or any((x in self.parameters for x in ('address', 'netmask', 'subnet_name'))):
            self.set_interface_type('ip')
        return
    protocol_types = set()
    unknown_protocols = []
    for protocol in protocols:
        if protocol.lower() in ['fc-nvme', 'fcp']:
            protocol_types.add('fc')
        elif protocol.lower() in ['nfs', 'cifs', 'iscsi']:
            protocol_types.add('ip')
        elif protocol.lower() != 'none':
            unknown_protocols.append(protocol)
    errors = []
    if unknown_protocols:
        errors.append('unexpected value(s) for protocols: %s' % unknown_protocols)
    if len(protocol_types) > 1:
        errors.append('incompatible value(s) for protocols: %s' % protocols)
    if errors:
        self.module.fail_json(msg='Error: unable to determine interface type, please set interface_type: %s' % ' - '.join(errors))
    if protocol_types:
        self.set_interface_type(protocol_types.pop())
    return