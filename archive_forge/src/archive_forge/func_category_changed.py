from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.ipa import IPAClient, ipa_argument_spec
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def category_changed(module, client, category_name, ipa_sudorule):
    if ipa_sudorule.get(category_name, None) == ['all']:
        if not module.check_mode:
            client.sudorule_mod(name=ipa_sudorule.get('cn')[0], item={category_name: None})
        return True
    return False