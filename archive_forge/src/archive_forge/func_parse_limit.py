from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from re import M, findall, search
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.argspec.firewall_rules.firewall_rules import (
def parse_limit(self, conf, attrib=None):
    """
        This function triggers the parsing of 'limit' attributes.
        :param conf: configuration to be parsed.
        :param attrib: 'limit'
        :return: generated config dictionary.
        """
    cfg_dict = self.parse_attr(conf, ['burst'], match=attrib)
    cfg_dict['rate'] = self.parse_rate(conf, 'rate')
    return cfg_dict