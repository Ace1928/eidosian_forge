from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.ipa import IPAClient, ipa_argument_spec
from ansible.module_utils.common.text.converters import to_native
def get_service_diff(client, ipa_host, module_service):
    non_updateable_keys = ['force', 'krbcanonicalname', 'skip_host_check']
    for key in non_updateable_keys:
        if key in module_service:
            del module_service[key]
    return client.get_diff(ipa_data=ipa_host, module_data=module_service)