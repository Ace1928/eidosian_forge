from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.facts import Facts
def delete_nested_dict(self, have, want):
    """
        Returns tlv_select nested dict that needs to be defaulted
        """
    outer_dict = {}
    for key, val in have.items():
        inner_dict = {}
        if not isinstance(val, dict):
            if key not in want.keys():
                inner_dict.update({key: val})
                return inner_dict
        elif key in want.keys():
            outer_dict.update({key: self.delete_nested_dict(val, want[key])})
        else:
            outer_dict.update({key: val})
    return outer_dict