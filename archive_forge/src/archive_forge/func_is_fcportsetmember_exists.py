from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible.module_utils._text import to_native
def is_fcportsetmember_exists(self):
    merged_result = {}
    cmd = 'lsfcportsetmember'
    cmdopts = {'filtervalue': 'portset_name={0}:fc_io_port_id={1}'.format(self.name, self.fcportid)}
    data = self.restapi.svc_run_command(cmd, cmdopts, cmdargs=None)
    if isinstance(data, list):
        for d in data:
            merged_result.update(d)
    else:
        merged_result = data
    self.fcportsetmember_details = merged_result
    return merged_result