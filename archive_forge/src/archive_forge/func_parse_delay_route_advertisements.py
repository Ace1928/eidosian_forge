from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils._text import to_bytes
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import string_types
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.junipernetworks.junos.plugins.module_utils.network.junos.argspec.bgp_address_family.bgp_address_family import (
def parse_delay_route_advertisements(self, cfg):
    """

        :param self:
        :param cfg:
        :return:
        """
    dra_dict = {}
    dra = cfg.get('delay-route-advertisements')
    if not dra:
        dra_dict['set'] = True
    else:
        if 'maximum-delay' in dra.keys():
            mxd = dra.get('maximum-delay')
            if 'route-age' in mxd.keys():
                dra_dict['max_delay_route_age'] = mxd.get('route-age')
            if 'routing-uptime' in mxd.keys():
                dra_dict['max_delay_routing_uptime'] = mxd.get('routing-uptime')
        if 'minimum-delay' in dra.keys():
            mid = dra.get('minimum-delay')
            if 'inbound-convergence' in mid.keys():
                dra_dict['min_delay_inbound_convergence'] = mid.get('inbound-convergence')
            if 'routing-uptime' in mid.keys():
                dra_dict['min_delay_routing_uptime'] = mid.get('routing-uptime')
    return dra_dict