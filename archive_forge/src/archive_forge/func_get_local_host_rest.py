from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, netapp_ipaddress
def get_local_host_rest(self):
    """
        Retrieves IP to hostname mapping for SVM of the cluster.
        """
    api = 'name-services/local-hosts'
    query = {'owner.name': self.parameters['owner'], 'address': self.parameters['address'], 'fields': 'address,hostname,owner.name,owner.uuid,aliases'}
    record, error = rest_generic.get_one_record(self.rest_api, api, query)
    if error:
        self.module.fail_json(msg='Error fetching IP to hostname mappings for %s: %s' % (self.parameters['owner'], to_native(error)), exception=traceback.format_exc())
    if record:
        self.owner_uuid = record['owner']['uuid']
        return {'address': self.na_helper.safe_get(record, ['address']), 'host': self.na_helper.safe_get(record, ['hostname']), 'aliases': self.na_helper.safe_get(record, ['aliases'])}
    return record