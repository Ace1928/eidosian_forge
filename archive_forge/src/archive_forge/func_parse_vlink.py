from __future__ import absolute_import, division, print_function
from copy import deepcopy
from re import M, findall, search
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.argspec.ospfv2.ospfv2 import (
def parse_vlink(self, conf):
    """
        This function triggers the parsing of 'virtual_link' attributes
        :param conf: configuration data
        :return: generated rule configuration dictionary
        """
    rule = self.parse_attrib(conf, 'vlink')
    r_sub = {'authentication': self.parse_authentication(conf, 'authentication')}
    rule.update(r_sub)
    return rule