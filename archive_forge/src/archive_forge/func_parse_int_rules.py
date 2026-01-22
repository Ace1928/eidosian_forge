from __future__ import absolute_import, division, print_function
from copy import deepcopy
from re import M, findall, search
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.argspec.firewall_interfaces.firewall_interfaces import (
def parse_int_rules(self, conf, afi):
    """
        This function forms the regex to fetch the 'access-rules'
        for specific interface based on ip-type.
        :param conf: configuration data.
        :param rules: rules configured per interface.
        :param afi: ip address type.
        :return: generated rule configuration dictionary.
        """
    r_lst = []
    config = {}
    rules = ['in', 'out', 'local']
    for r in set(rules):
        fr = {}
        r_regex = ' %s .+$' % r
        cfg = '\n'.join(findall(r_regex, conf, M))
        if cfg:
            fr = self.parse_rules(cfg, afi)
        else:
            out = search('^.*firewall ' + "'" + r + "'" + '(.*)', conf, M)
            if out:
                fr = {'direction': r}
        if fr:
            r_lst.append(fr)
    if r_lst:
        r_lst = sorted(r_lst, key=lambda i: i['direction'])
        config = {'afi': afi, 'rules': r_lst}
    return config