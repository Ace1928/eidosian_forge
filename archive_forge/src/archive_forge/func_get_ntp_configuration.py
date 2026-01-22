from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.ntp.ntp import NtpArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_ntp_configuration(self):
    """Get all NTP configuration"""
    all_ntp_request = [{'path': 'data/openconfig-system:system/ntp', 'method': GET}]
    all_ntp_response = []
    try:
        all_ntp_response = edit_config(self._module, to_request(self._module, all_ntp_request))
    except ConnectionError as exc:
        self._module.fail_json(msg=str(exc), code=exc.code)
    all_ntp_config = dict()
    if 'openconfig-system:ntp' in all_ntp_response[0][1]:
        all_ntp_config = all_ntp_response[0][1].get('openconfig-system:ntp', {})
    ntp_global_config = dict()
    if 'config' in all_ntp_config:
        ntp_global_config = all_ntp_config.get('config', {})
    ntp_servers = []
    if 'servers' in all_ntp_config:
        ntp_servers = all_ntp_config['servers'].get('server', [])
    ntp_keys = []
    if 'ntp-keys' in all_ntp_config:
        ntp_keys = all_ntp_config['ntp-keys'].get('ntp-key', [])
    ntp_config = dict()
    if 'network-instance' in ntp_global_config and ntp_global_config['network-instance']:
        ntp_config['vrf'] = ntp_global_config['network-instance']
    if 'enable-ntp-auth' in ntp_global_config:
        ntp_config['enable_ntp_auth'] = ntp_global_config['enable-ntp-auth']
    if 'source-interface' in ntp_global_config and ntp_global_config['source-interface']:
        ntp_config['source_interfaces'] = ntp_global_config['source-interface']
    if 'trusted-key' in ntp_global_config and ntp_global_config['trusted-key']:
        ntp_config['trusted_keys'] = ntp_global_config['trusted-key']
    servers = []
    for ntp_server in ntp_servers:
        if 'config' in ntp_server:
            server = {}
            server['address'] = ntp_server['config'].get('address', None)
            if 'key-id' in ntp_server['config']:
                server['key_id'] = ntp_server['config']['key-id']
            server['minpoll'] = ntp_server['config'].get('minpoll', None)
            server['maxpoll'] = ntp_server['config'].get('maxpoll', None)
            server['prefer'] = ntp_server['config'].get('prefer', None)
            servers.append(server)
    if servers:
        ntp_config['servers'] = servers
    keys = []
    for ntp_key in ntp_keys:
        if 'config' in ntp_key:
            key = {}
            key['encrypted'] = ntp_key['config'].get('encrypted', None)
            key['key_id'] = ntp_key['config'].get('key-id', None)
            key_type_str = ntp_key['config'].get('key-type', None)
            key_type = key_type_str.split(':', 1)[-1]
            key['key_type'] = key_type
            key['key_value'] = ntp_key['config'].get('key-value', None)
            keys.append(key)
    if keys:
        ntp_config['ntp_keys'] = keys
    return ntp_config