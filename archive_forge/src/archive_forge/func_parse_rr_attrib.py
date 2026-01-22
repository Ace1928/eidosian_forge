from __future__ import absolute_import, division, print_function
from copy import deepcopy
from re import M, findall, search
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.argspec.firewall_global.firewall_global import (
def parse_rr_attrib(self, conf, attrib=None):
    """
        This function fetches the 'ip_src_route'
        invoke function to parse icmp redirects.
        :param conf: configuration to be parsed.
        :param attrib: 'ipv4/ipv6'.
        :return: generated config dictionary.
        """
    cfg_dict = self.parse_attr(conf, ['ip_src_route'], type=attrib)
    cfg_dict['icmp_redirects'] = self.parse_icmp_redirects(conf, attrib)
    cfg_dict['afi'] = attrib
    return cfg_dict