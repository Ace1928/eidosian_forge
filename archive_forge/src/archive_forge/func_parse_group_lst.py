from __future__ import absolute_import, division, print_function
from copy import deepcopy
from re import M, findall, search
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.argspec.firewall_global.firewall_global import (
def parse_group_lst(self, conf, type, include_afi=True):
    """
        This function fetches the name of group and invoke function to
        parse group attributes'.
        :param conf: configuration data.
        :param type: type of group.
        :param include_afi: if the afi should be included in the parsed object
        :return: generated group list configuration.
        """
    g_lst = []
    groups = findall('^set firewall group ' + type + ' (\\S+)', conf, M)
    if groups:
        rules_lst = []
        for gr in set(groups):
            gr_regex = ' %s .+$' % gr
            cfg = '\n'.join(findall(gr_regex, conf, M))
            if 'ipv6' in type:
                obj = self.parse_groups(cfg, type[len('ipv6-'):], gr)
                if include_afi:
                    obj['afi'] = 'ipv6'
            else:
                obj = self.parse_groups(cfg, type, gr)
                if include_afi:
                    obj['afi'] = 'ipv4'
            obj['name'] = gr.strip("'")
            if obj:
                rules_lst.append(obj)
        g_lst = sorted(rules_lst, key=lambda i: i['name'])
    return g_lst