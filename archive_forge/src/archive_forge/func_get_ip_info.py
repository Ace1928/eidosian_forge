from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
def get_ip_info(self):
    all_data = self.restapi.svc_obj_info(cmd='lsip', cmdopts=None, cmdargs=None)
    if self.portset:
        data = list(filter(lambda item: item['node_name'] == self.node and item['port_id'] == str(self.port) and (item['portset_name'] == self.portset) and (item['IP_address'] == self.ip_address), all_data))
    else:
        data = list(filter(lambda item: item['node_name'] == self.node and item['port_id'] == str(self.port) and (item['IP_address'] == self.ip_address), all_data))
        if len(data) > 1:
            self.module.fail_json(msg='Module could not find the exact IP with [node, port, ip_address]. Please also use [portset].')
    self.log('GET: IP data: %s', data)
    return data