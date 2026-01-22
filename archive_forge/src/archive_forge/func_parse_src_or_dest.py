from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from re import M, findall, search
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.argspec.firewall_rules.firewall_rules import (
def parse_src_or_dest(self, conf, attrib=None):
    """
        This function triggers the parsing of 'source or
        destination' attributes.
        :param conf: configuration.
        :param attrib:'source/destination'.
        :return:generated source/destination configuration dictionary.
        """
    a_lst = ['port', 'address', 'mac_address']
    cfg_dict = self.parse_attr(conf, a_lst, match=attrib)
    cfg_dict['group'] = self.parse_group(conf, attrib + ' group')
    return cfg_dict