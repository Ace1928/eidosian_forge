from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.ipa import IPAClient, ipa_argument_spec
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def get_subca_diff(client, ipa_subca, module_subca):
    details = get_subca_dict(module_subca)
    return client.get_diff(ipa_data=ipa_subca, module_data=details)