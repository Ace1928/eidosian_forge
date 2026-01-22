from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible.module_utils.connection import ConnectionError
def build_create_evpn_payload(self, conf):
    evpn_nvo_list = [{'name': conf['evpn_nvo'], 'source_vtep': conf['name']}]
    evpn_dict = {'sonic-vxlan:EVPN_NVO_LIST': evpn_nvo_list}
    return evpn_dict