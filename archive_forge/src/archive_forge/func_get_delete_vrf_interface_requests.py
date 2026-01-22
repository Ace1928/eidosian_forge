from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
from ansible.module_utils.connection import ConnectionError
def get_delete_vrf_interface_requests(self, configs, have, state=None):
    requests = []
    if not configs:
        return requests
    method = DELETE
    for conf in configs:
        name = conf['name']
        empty_flag = False
        members = conf.get('members', None)
        if members:
            interfaces = members.get('interfaces', None)
        if members is None:
            empty_flag = True
        elif members is not None and interfaces is None:
            empty_flag = True
        matched = next((have_cfg for have_cfg in have if have_cfg['name'] == name), None)
        if not matched:
            continue
        adjusted_delete_all_flag = name != MGMT_VRF_NAME and self.delete_all_flag
        adjusted_empty_flag = empty_flag
        if state == 'replaced':
            adjusted_empty_flag = empty_flag and name != MGMT_VRF_NAME
        if adjusted_delete_all_flag or adjusted_empty_flag:
            url = 'data/openconfig-network-instance:network-instances/network-instance={0}'.format(name)
            request = {'path': url, 'method': method}
            requests.append(request)
        else:
            have_members = matched.get('members', None)
            conf_members = conf.get('members', None)
            if have_members:
                have_intf = have_members.get('interfaces', None)
                conf_intf = conf_members.get('interfaces', None)
                if conf_intf:
                    for del_mem in conf_intf:
                        if del_mem in have_intf:
                            url = 'data/openconfig-network-instance:network-instances/'
                            url = url + 'network-instance={0}/interfaces/interface={1}'.format(name, del_mem['name'])
                            request = {'path': url, 'method': method}
                            requests.append(request)
    return requests