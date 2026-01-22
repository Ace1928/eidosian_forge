from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import \
from ansible.module_utils._text import to_native
def get_existing_vg(self):
    merged_result = {}
    data = self.restapi.svc_obj_info(cmd='lsvolumegroup', cmdopts=None, cmdargs=['-gui', self.name])
    if isinstance(data, list):
        for d in data:
            merged_result.update(d)
    else:
        merged_result = data
    if merged_result and (self.snapshotpolicy and self.policystarttime or self.snapshotpolicysuspended):
        SP_data = self.restapi.svc_obj_info(cmd='lsvolumegroupsnapshotpolicy', cmdopts=None, cmdargs=[self.name])
        merged_result['snapshot_policy_start_time'] = SP_data['snapshot_policy_start_time']
        merged_result['snapshot_policy_suspended'] = SP_data['snapshot_policy_suspended']
    return merged_result