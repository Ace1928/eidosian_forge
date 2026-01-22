from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.ipa import IPAClient, ipa_argument_spec
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def get_subca_dict(details=None):
    module_subca = dict()
    if details['description'] is not None:
        module_subca['description'] = details['description']
    if details['subca_subject'] is not None:
        module_subca['ipacasubjectdn'] = details['subca_subject']
    return module_subca