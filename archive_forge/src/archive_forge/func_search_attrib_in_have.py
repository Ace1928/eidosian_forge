from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.utils.utils import (
def search_attrib_in_have(self, have, want, attr):
    """
        This function  returns the attribute if it is present in target config.
        :param have: the target config.
        :param want: the desired config.
        :param attr: attribute name .
        :return: attribute/None
        """
    if have:
        for h in have:
            if h[attr] == want[attr]:
                return h
    return None