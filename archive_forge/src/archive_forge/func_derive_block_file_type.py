from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, netapp_ipaddress
def derive_block_file_type(self, protocols):
    block_p, file_p, fcp = (False, False, False)
    if protocols is None:
        fcp = self.parameters.get('interface_type') == 'fc'
        return (fcp, file_p, fcp)
    block_values, file_values = ([], [])
    for protocol in protocols:
        if protocol.lower() in ['fc-nvme', 'fcp', 'iscsi']:
            block_p = True
            block_values.append(protocol)
            if protocol.lower() in ['fc-nvme', 'fcp']:
                fcp = True
        elif protocol.lower() in ['nfs', 'cifs']:
            file_p = True
            file_values.append(protocol)
    if block_p and file_p:
        self.module.fail_json(msg='Cannot use any of %s with %s' % (block_values, file_values))
    return (block_p, file_p, fcp)