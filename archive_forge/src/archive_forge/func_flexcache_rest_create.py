from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_volume
def flexcache_rest_create(self):
    """ use POST to create a FlexCache """
    mappings = dict(name='name', vserver='svm.name', junction_path='path', size='size', aggr_list='aggregates', aggr_list_multiplier='constituents_per_aggregate', origins='origins', prepopulate='prepopulate')
    body = self.flexcache_rest_create_body(mappings)
    api = 'storage/flexcache/flexcaches'
    response, error = rest_generic.post_async(self.rest_api, api, body, job_timeout=self.parameters['time_out'])
    self.na_helper.fail_on_error(error)
    return response