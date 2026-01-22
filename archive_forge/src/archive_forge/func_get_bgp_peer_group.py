from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, netapp_ipaddress
def get_bgp_peer_group(self, name=None):
    """
        Get BGP peer group.
        """
    if name is None:
        name = self.parameters['name']
    api = 'network/ip/bgp/peer-groups'
    query = {'name': name, 'fields': 'name,uuid,peer'}
    if 'ipspace' in self.parameters:
        query['ipspace.name'] = self.parameters['ipspace']
    record, error = rest_generic.get_one_record(self.rest_api, api, query)
    if error:
        self.module.fail_json(msg='Error fetching BGP peer group %s: %s' % (name, to_native(error)), exception=traceback.format_exc())
    if record:
        self.uuid = record['uuid']
        return {'name': self.na_helper.safe_get(record, ['name']), 'peer': self.na_helper.safe_get(record, ['peer'])}
    return None