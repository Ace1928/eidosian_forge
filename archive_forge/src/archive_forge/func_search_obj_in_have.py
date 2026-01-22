from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.utils.utils import (
def search_obj_in_have(self, have, w_name, key):
    """
        This function  returns the rule-set/rule if it is present in target config.
        :param have: target config.
        :param w_name: rule-set name.
        :param type: rule_sets/rule/r_list.
        :return: rule-set/rule.
        """
    if have:
        for item in have:
            if item[key] == w_name[key]:
                return item
    return None