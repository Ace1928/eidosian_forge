from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
def get_dns_rest(self):
    if not self.parameters.get('vserver') and (not self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 9, 1)):
        return self.get_cluster_dns()
    api = 'name-services/dns'
    params = {'fields': 'domains,servers,svm'}
    if self.parameters.get('vserver'):
        params['svm.name'] = self.parameters['vserver']
    else:
        params['scope'] = 'cluster'
    record, error = rest_generic.get_one_record(self.rest_api, api, params)
    if error:
        self.module.fail_json(msg='Error getting DNS service: %s' % error)
    if record:
        if params.get('scope') == 'cluster':
            uuid = record.get('uuid')
        else:
            uuid = self.na_helper.safe_get(record, ['svm', 'uuid'])
        return {'domains': record.get('domains'), 'nameservers': record.get('servers'), 'uuid': uuid}
    if self.parameters.get('vserver') and (not self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 9, 1)):
        return self.get_cluster_dns()
    return None