from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.facts import Facts
def find_delete_params(self, have, want):
    """
        Returns parameters that are present in have and not in want, that need to be defaulted
        """
    delete_dict = {}
    for key, val in have.items():
        if key not in want.keys():
            delete_dict.update({key: val})
        elif key == 'tlv_select':
            delete_dict.update({key: self.delete_nested_dict(have['tlv_select'], want['tlv_select'])})
    return delete_dict