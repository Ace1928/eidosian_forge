from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def get_initiator(self):
    """
        Get current initiator.
        :return: dict of current initiator details.
        """
    params = {'fields': '*', 'initiator': self.parameters['initiator']}
    api = 'protocols/san/iscsi/credentials'
    message, error = self.rest_api.get(api, params)
    if error is not None:
        self.module.fail_json(msg='Error on fetching initiator: %s' % error)
    if message['num_records'] > 0:
        record = message['records'][0]
        initiator_details = {'auth_type': record['authentication_type']}
        if initiator_details['auth_type'] == 'chap':
            if record['chap'].get('inbound'):
                initiator_details['inbound_username'] = record['chap']['inbound']['user']
            else:
                initiator_details['inbound_username'] = None
            if record['chap'].get('outbound'):
                initiator_details['outbound_username'] = record['chap']['outbound']['user']
            else:
                initiator_details['outbound_username'] = None
        if record.get('initiator_address'):
            if record['initiator_address'].get('ranges'):
                ranges = []
                for address_range in record['initiator_address']['ranges']:
                    if address_range['start'] == address_range['end']:
                        ranges.append(address_range['start'])
                    else:
                        ranges.append(address_range['start'] + '-' + address_range['end'])
                initiator_details['address_ranges'] = ranges
            else:
                initiator_details['address_ranges'] = []
        else:
            initiator_details['address_ranges'] = []
        return initiator_details