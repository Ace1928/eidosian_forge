from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.rest_application import RestApplication
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
def validate_snaplock_changes(self, current, modify=None, after_create=False):
    if not self.use_rest:
        return
    msg = None
    if modify:
        if 'type' in modify['snaplock']:
            msg = 'Error: volume snaplock type was not set properly at creation time.' if after_create else 'Error: changing a volume snaplock type after creation is not allowed.'
            msg += '  Current: %s, desired: %s.' % (current['snaplock']['type'], self.parameters['snaplock']['type'])
    elif self.parameters['state'] == 'present':
        sl_dict = self.na_helper.filter_out_none_entries(self.parameters.get('snaplock', {}))
        sl_type = sl_dict.pop('type', 'non_snaplock')
        if sl_dict and (current is None and sl_type == 'non_snaplock' or (current and current['snaplock']['type'] == 'non_snaplock')):
            msg = 'Error: snaplock options are not supported for non_snaplock volume, found: %s.' % sl_dict
        if not self.rest_api.meets_rest_minimum_version(True, 9, 10, 1):
            if sl_type == 'non_snaplock':
                self.parameters.pop('snaplock', None)
            else:
                msg = 'Error: %s' % self.rest_api.options_require_ontap_version('snaplock type', '9.10.1', True)
    if msg:
        self.module.fail_json(msg=msg)