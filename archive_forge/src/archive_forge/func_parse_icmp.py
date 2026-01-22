from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from re import M, findall, search
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.argspec.firewall_rules.firewall_rules import (
def parse_icmp(self, conf, attrib=None):
    """
        This function triggers the parsing of 'icmp' attributes.
        :param conf: configuration to be parsed.
        :param attrib: 'icmp'.
        :return: generated config dictionary.
        """
    a_lst = ['code', 'type', 'type_name']
    if attrib == 'icmp':
        attrib = 'icmpv6'
    conf = re.sub('icmpv6 type', 'icmpv6 type-name', conf)
    cfg_dict = self.parse_attr(conf, a_lst, match=attrib)
    return cfg_dict