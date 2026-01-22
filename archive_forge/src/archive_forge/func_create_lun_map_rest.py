from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
import codecs
from ansible.module_utils._text import to_text, to_bytes
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_lun_map_rest(self):
    api = 'protocols/san/lun-maps'
    body = {'svm.name': self.parameters['vserver'], 'igroup.name': self.parameters['initiator_group_name'], 'lun.name': self.parameters['path']}
    if self.parameters.get('lun_id') is not None:
        body['logical_unit_number'] = self.parameters['lun_id']
    dummy, error = rest_generic.post_async(self.rest_api, api, body, job_timeout=120)
    if error:
        self.module.fail_json(msg='Error creating lun_map %s: %s' % (self.parameters['path'], to_native(error)), exception=traceback.format_exc())