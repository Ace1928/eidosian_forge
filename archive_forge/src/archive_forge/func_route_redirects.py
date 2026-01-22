from __future__ import absolute_import, division, print_function
from copy import deepcopy
from re import M, findall, search
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.argspec.firewall_global.firewall_global import (
def route_redirects(self, conf):
    """
        This function forms the regex to fetch the afi and invoke
        functions to fetch route redirects and source routes
        :param conf: configuration data.
        :return: generated rule list configuration.
        """
    rr_lst = []
    v6_attr = findall('^set firewall (?:ipv6-src-route|ipv6-receive-redirects) (\\S+)', conf, M)
    if v6_attr:
        obj = self.parse_rr_attrib(conf, 'ipv6')
        if obj:
            rr_lst.append(obj)
    v4_attr = findall('^set firewall (?:ip-src-route|receive-redirects|send-redirects) (\\S+)', conf, M)
    if v4_attr:
        obj = self.parse_rr_attrib(conf, 'ipv4')
        if obj:
            rr_lst.append(obj)
    return rr_lst