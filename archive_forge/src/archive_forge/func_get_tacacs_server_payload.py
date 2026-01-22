from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible.module_utils.connection import ConnectionError
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
def get_tacacs_server_payload(self, hosts):
    payload = {}
    servers_load = []
    for host in hosts:
        if host.get('name', None):
            host_cfg = {'address': host['name']}
            if host.get('auth_type', None):
                host_cfg['auth-type'] = host['auth_type']
            if host.get('priority', None):
                host_cfg['priority'] = host['priority']
            if host.get('vrf', None):
                host_cfg['vrf'] = host['vrf']
            if host.get('timeout', None):
                host_cfg['timeout'] = host['timeout']
            tacacs_port_key_cfg = {}
            if host.get('port', None):
                tacacs_port_key_cfg['port'] = host['port']
            if host.get('key', None):
                tacacs_port_key_cfg['secret-key'] = host['key']
            if tacacs_port_key_cfg:
                consolidated_load = {'address': host['name']}
                consolidated_load['config'] = host_cfg
                consolidated_load['tacacs'] = {'config': tacacs_port_key_cfg}
                servers_load.append(consolidated_load)
    if servers_load:
        payload = {'openconfig-system:servers': {'server': servers_load}}
    return payload