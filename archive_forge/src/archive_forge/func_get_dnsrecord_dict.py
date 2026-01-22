from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.ipa import IPAClient, ipa_argument_spec
from ansible.module_utils.common.text.converters import to_native
def get_dnsrecord_dict(details=None):
    module_dnsrecord = dict()
    if details['record_type'] == 'A' and details['record_values']:
        module_dnsrecord.update(arecord=details['record_values'])
    elif details['record_type'] == 'AAAA' and details['record_values']:
        module_dnsrecord.update(aaaarecord=details['record_values'])
    elif details['record_type'] == 'A6' and details['record_values']:
        module_dnsrecord.update(a6record=details['record_values'])
    elif details['record_type'] == 'CNAME' and details['record_values']:
        module_dnsrecord.update(cnamerecord=details['record_values'])
    elif details['record_type'] == 'DNAME' and details['record_values']:
        module_dnsrecord.update(dnamerecord=details['record_values'])
    elif details['record_type'] == 'NS' and details['record_values']:
        module_dnsrecord.update(nsrecord=details['record_values'])
    elif details['record_type'] == 'PTR' and details['record_values']:
        module_dnsrecord.update(ptrrecord=details['record_values'])
    elif details['record_type'] == 'TXT' and details['record_values']:
        module_dnsrecord.update(txtrecord=details['record_values'])
    elif details['record_type'] == 'SRV' and details['record_values']:
        module_dnsrecord.update(srvrecord=details['record_values'])
    elif details['record_type'] == 'MX' and details['record_values']:
        module_dnsrecord.update(mxrecord=details['record_values'])
    if details.get('record_ttl'):
        module_dnsrecord.update(dnsttl=details['record_ttl'])
    return module_dnsrecord