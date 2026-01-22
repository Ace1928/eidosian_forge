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
def delete_lun_map_rest(self):
    api = 'protocols/san/lun-maps'
    both_uuids = '%s/%s' % (self.lun_uuid, self.igroup_uuid)
    dummy, error = rest_generic.delete_async(self.rest_api, api, both_uuids, job_timeout=120)
    if error:
        self.module.fail_json(msg='Error deleting lun_map %s: %s' % (self.parameters['path'], to_native(error)), exception=traceback.format_exc())