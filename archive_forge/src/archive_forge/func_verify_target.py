from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
def verify_target(self):
    self.log('Entering function verify_target()')
    source_data, target_data = self.get_existing_vdisk()
    if source_data:
        if source_data[0]['RC_name']:
            self.module.fail_json(msg='Source Volume [%s] is already in a relationship.' % self.source_volume)
    if target_data:
        if target_data[0]['RC_name']:
            self.module.fail_json(msg='Target Volume [%s] is already in a relationship.' % self.target_volume)
        if target_data[0]['mdisk_grp_name'] != self.remote_pool:
            self.module.fail_json(msg='Target Volume [%s] exists on a different pool.' % self.target_volume)
    if not source_data:
        self.module.fail_json(msg='Source Volume [%s] does not exist.' % self.source_volume)
    elif source_data and target_data:
        source_size = int(source_data[0]['capacity'])
        remote_size = int(target_data[0]['capacity'])
        if source_size != remote_size:
            self.module.fail_json(msg='Remote Volume size is different than that of source volume.')
        else:
            self.log('Target volume already exists, verifying volume mappings now..')
            self.verify_remote_volume_mapping()
    elif source_data and (not target_data):
        self.vdisk_create(source_data)
        self.log('Target volume successfully created')
        self.changed = True