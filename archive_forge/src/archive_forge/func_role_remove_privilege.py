from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.ipa import IPAClient, ipa_argument_spec
from ansible.module_utils.common.text.converters import to_native
def role_remove_privilege(self, name, item):
    return self._post_json(method='role_remove_privilege', name=name, item={'privilege': item})