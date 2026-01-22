from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_subnet_rest(self, name):
    api = 'network/ip/subnets'
    params = {'name': name, 'fields': 'available_ip_ranges,name,broadcast_domain,ipspace,gateway,subnet,uuid'}
    if self.parameters.get('ipspace'):
        params['ipspace.name'] = self.parameters['ipspace']
    record, error = rest_generic.get_one_record(self.rest_api, api, params)
    if error:
        self.module.fail_json(msg='Error fetching subnet %s: %s' % (name, error))
    current = None
    if record:
        self.uuid = record['uuid']
        current = {'name': record['name'], 'broadcast_domain': self.na_helper.safe_get(record, ['broadcast_domain', 'name']), 'gateway': self.na_helper.safe_get(record, ['gateway']), 'ipspace': self.na_helper.safe_get(record, ['ipspace', 'name']), 'subnet': record['subnet']['address'] + '/' + record['subnet']['netmask'], 'ip_ranges': []}
        for each_range in record.get('available_ip_ranges', []):
            if each_range['start'] == each_range['end']:
                current['ip_ranges'].append(each_range['start'])
            else:
                current['ip_ranges'].append(each_range['start'] + '-' + each_range['end'])
    return current