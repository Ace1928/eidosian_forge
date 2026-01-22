from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.ipa import IPAClient, ipa_argument_spec
from ansible.module_utils.common.text.converters import to_native
def dnsrecord_mod(self, zone_name=None, record_name=None, details=None):
    item = get_dnsrecord_dict(details)
    item.update(idnsname=record_name)
    if details.get('record_ttl'):
        item.update(dnsttl=details['record_ttl'])
    return self._post_json(method='dnsrecord_mod', name=zone_name, item=item)