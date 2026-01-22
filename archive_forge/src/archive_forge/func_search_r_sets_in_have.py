from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.utils.utils import (
def search_r_sets_in_have(self, have, w_name, type='rule_sets', afi=None):
    """
        This function  returns the rule-set/rule if it is present in target config.
        :param have: target config.
        :param w_name: rule-set name.
        :param type: rule_sets/rule/r_list.
        :param afi: address family (when type is r_list).
        :return: rule-set/rule.
        """
    if have:
        key = 'name'
        if type == 'rules':
            key = 'number'
            for r in have:
                if r[key] == w_name:
                    return r
        elif type == 'r_list':
            for h in have:
                if h['afi'] == afi:
                    r_sets = self._get_r_sets(h)
                    for rs in r_sets:
                        if rs[key] == w_name:
                            return rs
        else:
            for rs in have:
                if rs[key] == w_name:
                    return rs
    return None