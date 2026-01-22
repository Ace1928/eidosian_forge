from __future__ import absolute_import, division, print_function
from copy import deepcopy
from re import M, findall, search
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.argspec.ospfv2.ospfv2 import (
def parse_leaf_list(self, conf, attrib):
    """
        This function forms the regex to fetch the listed attributes
        from the configuration data
        :param conf: configuration data
        :param attrib: attribute name
        :return: generated rule list configuration
        """
    lst = []
    items = findall('^' + attrib + " (?:'*)(\\S+)(?:'*)", conf, M)
    if items:
        for i in set(items):
            lst.append(i.strip("'"))
            lst.sort()
    return lst