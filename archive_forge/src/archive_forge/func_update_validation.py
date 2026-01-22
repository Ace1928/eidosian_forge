from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import \
from ansible.module_utils._text import to_native
def update_validation(self, data):
    mutually_exclusive = (('ownershipgroup', 'noownershipgroup'), ('safeguardpolicyname', 'nosafeguardpolicy'), ('ownershipgroup', 'safeguardpolicyname'), ('ownershipgroup', 'snapshotpolicy'), ('ownershipgroup', 'policystarttime'), ('nosafeguardpolicy', 'nosnapshotpolicy'), ('snapshotpolicy', 'nosnapshotpolicy'), ('snapshotpolicy', 'safeguardpolicyname'), ('replicationpolicy', 'noreplicationpolicy'))
    for param1, param2 in mutually_exclusive:
        if getattr(self, param1) and getattr(self, param2):
            self.module.fail_json(msg='Mutually exclusive parameters: {0}, {1}'.format(param1, param2))
    unsupported_maps = (('type', data.get('volume_group_type', '')), ('snapshot', data.get('source_snapshot', '')), ('fromsourcegroup', data.get('source_volume_group_name', '')))
    unsupported = (fields[0] for fields in unsupported_maps if getattr(self, fields[0]) and getattr(self, fields[0]) != fields[1])
    unsupported_exists = ', '.join(unsupported)
    if unsupported_exists:
        self.module.fail_json(msg='Following paramters not supported during update: {0}'.format(unsupported_exists))