from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.ipa import IPAClient, ipa_argument_spec
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def subca_mod(self, subca_name=None, diff=None, details=None):
    item = get_subca_dict(details)
    for change in diff:
        update_detail = dict()
        if item[change] is not None:
            update_detail.update(setattr='{0}={1}'.format(change, item[change]))
            self._post_json(method='ca_mod', name=subca_name, item=update_detail)