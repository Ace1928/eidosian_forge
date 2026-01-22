from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
def validate_async_options(self):
    if 'policy_type' in self.parameters:
        disallowed_options = {'vault': ['copy_latest_source_snapshot', 'copy_all_source_snapshots'], 'mirror-vault': ['copy_latest_source_snapshot', 'copy_all_source_snapshots', 'create_snapshot_on_source'], 'async_mirror': ['create_snapshot_on_source'], 'async': [], 'sync': ['copy_latest_source_snapshot', 'copy_all_source_snapshots', 'create_snapshot_on_source']}
        try:
            options = disallowed_options[self.parameters['policy_type']]
        except KeyError:
            options = disallowed_options['sync']
        for option in options:
            if option in self.parameters:
                self.fail_invalid_option(self.parameters['policy_type'], option)
    if self.use_rest:
        if 'copy_all_source_snapshots' in self.parameters and self.parameters.get('copy_all_source_snapshots') is not True:
            self.module.fail_json(msg='Error: the property copy_all_source_snapshots can only be set to true when present')
        if 'copy_latest_source_snapshot' in self.parameters and self.parameters.get('copy_latest_source_snapshot') is not True:
            self.module.fail_json(msg='Error: the property copy_latest_source_snapshot can only be set to true when present')
        if 'create_snapshot_on_source' in self.parameters and self.parameters['create_snapshot_on_source'] is not False:
            self.module.fail_json(msg='Error: the property create_snapshot_on_source can only be set to false when present')