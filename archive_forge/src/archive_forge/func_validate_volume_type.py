from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible.module_utils._text import to_native
def validate_volume_type(self, data):
    unsupported_volume = False
    if data[0]['type'] == 'many':
        unsupported_volume = True
    if not unsupported_volume:
        relationship_name = data[0]['RC_name']
        if relationship_name:
            rel_data = self.restapi.svc_obj_info(cmd='lsrcrelationship', cmdopts=None, cmdargs=[relationship_name])
            if rel_data['copy_type'] == 'activeactive':
                unsupported_volume = True
    if unsupported_volume:
        self.module.fail_json(msg='The module cannot be used for managing Mirrored volume.')