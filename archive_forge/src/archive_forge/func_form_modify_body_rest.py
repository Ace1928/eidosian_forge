from __future__ import absolute_import, division, print_function
import copy
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def form_modify_body_rest(self, modify, current):
    if modify.get('enabled') == 'disabled':
        return {'efficiency': {'dedupe': 'none', 'compression': 'none', 'compaction': 'none', 'cross_volume_dedupe': 'none'}}
    body = {}
    if modify.get('enabled') == 'enabled':
        body['efficiency.dedupe'] = 'background'
    if 'enable_compression' in modify or 'enable_inline_compression' in modify:
        body['efficiency.compression'] = self.derive_efficiency_type(modify.get('enable_compression'), modify.get('enable_inline_compression'), current.get('enable_compression'), current.get('enable_inline_compression'))
    if 'enable_cross_volume_background_dedupe' in modify or 'enable_cross_volume_inline_dedupe' in modify:
        body['efficiency.cross_volume_dedupe'] = self.derive_efficiency_type(modify.get('enable_cross_volume_background_dedupe'), modify.get('enable_cross_volume_inline_dedupe'), current.get('enable_cross_volume_background_dedupe'), current.get('enable_cross_volume_inline_dedupe'))
    if modify.get('enable_data_compaction'):
        body['efficiency.compaction'] = 'inline'
    elif modify.get('enable_data_compaction') is False:
        body['efficiency.compaction'] = 'none'
    if modify.get('enable_inline_dedupe'):
        body['efficiency.dedupe'] = 'both'
    elif modify.get('enable_inline_dedupe') is False:
        body['efficiency.dedupe'] = 'background'
    if self.parameters.get('policy'):
        body['efficiency.policy.name'] = self.parameters['policy']
    if modify.get('storage_efficiency_mode'):
        body['storage_efficiency_mode'] = modify['storage_efficiency_mode']
    if modify.get('status'):
        body['efficiency.scanner.state'] = modify['status']
    if 'start_ve_scan_old_data' in self.parameters:
        body['efficiency.scanner.scan_old_data'] = self.parameters['start_ve_scan_old_data']
    return body