from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.bgp_utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import to_request
from ansible.module_utils.connection import ConnectionError
from copy import deepcopy
def delete_specific_peergroup_param_request(self, vrf_name, cmd):
    requests = []
    delete_static_path = '%s=%s/%s' % (self.network_instance_path, vrf_name, self.protocol_bgp_path)
    delete_static_path = delete_static_path + '/peer-groups/peer-group=%s' % cmd['name']
    if cmd.get('remote_as', None) is not None:
        if cmd['remote_as'].get('peer_as', None) is not None:
            delete_path = delete_static_path + '/config/peer-as'
            requests.append({'path': delete_path, 'method': DELETE})
        elif cmd['remote_as'].get('peer_type', None) is not None:
            delete_path = delete_static_path + '/config/peer-type'
            requests.append({'path': delete_path, 'method': DELETE})
    if cmd.get('advertisement_interval', None) is not None:
        delete_path = delete_static_path + '/timers/config/minimum-advertisement-interval'
        requests.append({'path': delete_path, 'method': DELETE})
    if cmd.get('timers', None) is not None:
        if cmd['timers'].get('holdtime', None) is not None:
            delete_path = delete_static_path + '/timers/config/hold-time'
            requests.append({'path': delete_path, 'method': DELETE})
        if cmd['timers'].get('keepalive', None) is not None:
            delete_path = delete_static_path + '/timers/config/keepalive-interval'
            requests.append({'path': delete_path, 'method': DELETE})
        if cmd['timers'].get('connect_retry', None) is not None:
            delete_path = delete_static_path + '/timers/config/connect-retry'
            requests.append({'path': delete_path, 'method': DELETE})
    if cmd.get('capability', None) is not None:
        if cmd['capability'].get('dynamic', None) is not None:
            delete_path = delete_static_path + '/config/capability-dynamic'
            requests.append({'path': delete_path, 'method': DELETE})
        if cmd['capability'].get('extended_nexthop', None) is not None:
            delete_path = delete_static_path + '/config/capability-extended-nexthop'
            requests.append({'path': delete_path, 'method': DELETE})
    if cmd.get('pg_description', None) is not None:
        delete_path = delete_static_path + '/config/description'
        requests.append({'path': delete_path, 'method': DELETE})
    if cmd.get('disable_connected_check', None) is not None:
        delete_path = delete_static_path + '/config/disable-ebgp-connected-route-check'
        requests.append({'path': delete_path, 'method': DELETE})
    if cmd.get('dont_negotiate_capability', None) is not None:
        delete_path = delete_static_path + '/config/dont-negotiate-capability'
        requests.append({'path': delete_path, 'method': DELETE})
    if cmd.get('enforce_first_as', None) is not None:
        delete_path = delete_static_path + '/config/enforce-first-as'
        requests.append({'path': delete_path, 'method': DELETE})
    if cmd.get('enforce_multihop', None) is not None:
        delete_path = delete_static_path + '/config/enforce-multihop'
        requests.append({'path': delete_path, 'method': DELETE})
    if cmd.get('override_capability', None) is not None:
        delete_path = delete_static_path + '/config/override-capability'
        requests.append({'path': delete_path, 'method': DELETE})
    if cmd.get('shutdown_msg', None) is not None:
        delete_path = delete_static_path + '/config/shutdown-message'
        requests.append({'path': delete_path, 'method': DELETE})
    if cmd.get('solo', None) is not None:
        delete_path = delete_static_path + '/config/solo-peer'
        requests.append({'path': delete_path, 'method': DELETE})
    if cmd.get('strict_capability_match', None) is not None:
        delete_path = delete_static_path + '/config/strict-capability-match'
        requests.append({'path': delete_path, 'method': DELETE})
    if cmd.get('ttl_security', None) is not None:
        delete_path = delete_static_path + '/config/ttl-security-hops'
        requests.append({'path': delete_path, 'method': DELETE})
    if cmd.get('local_as', None) is not None:
        if cmd['local_as'].get('as', None) is not None:
            delete_path = delete_static_path + '/config/local-as'
            requests.append({'path': delete_path, 'method': DELETE})
        if cmd['local_as'].get('no_prepend', None) is not None:
            delete_path = delete_static_path + '/config/local-as-no-prepend'
            requests.append({'path': delete_path, 'method': DELETE})
        if cmd['local_as'].get('replace_as', None) is not None:
            delete_path = delete_static_path + '/config/local-as-replace-as'
            requests.append({'path': delete_path, 'method': DELETE})
    if cmd.get('local_address', None) is not None:
        delete_path = delete_static_path + '/transport/config/local-address'
        requests.append({'path': delete_path, 'method': DELETE})
    if cmd.get('passive', None) is not None:
        delete_path = delete_static_path + '/transport/config/passive-mode'
        requests.append({'path': delete_path, 'method': DELETE})
    if cmd.get('bfd', None) is not None:
        if cmd['bfd'].get('enabled', None) is not None:
            delete_path = delete_static_path + '/enable-bfd/config/enabled'
            requests.append({'path': delete_path, 'method': DELETE})
        if cmd['bfd'].get('check_failure', None) is not None:
            delete_path = delete_static_path + '/enable-bfd/config/check-control-plane-failure'
            requests.append({'path': delete_path, 'method': DELETE})
        if cmd['bfd'].get('profile', None) is not None:
            delete_path = delete_static_path + '/enable-bfd/config/bfd-profile'
            requests.append({'path': delete_path, 'method': DELETE})
    if cmd.get('auth_pwd', None) is not None:
        if cmd['auth_pwd'].get('pwd', None) is not None:
            delete_path = delete_static_path + '/auth-password/config/password'
            requests.append({'path': delete_path, 'method': DELETE})
        if cmd['auth_pwd'].get('encrypted', None) is not None:
            delete_path = delete_static_path + '/auth-password/config/encrypted'
            requests.append({'path': delete_path, 'method': DELETE})
    if cmd.get('ebgp_multihop', None) is not None:
        if cmd['ebgp_multihop'].get('enabled', None) is not None:
            delete_path = delete_static_path + '/ebgp-multihop/config/enabled'
            requests.append({'path': delete_path, 'method': DELETE})
        if cmd['ebgp_multihop'].get('multihop_ttl', None) is not None:
            delete_path = delete_static_path + '/ebgp-multihop/config/multihop-ttl'
            requests.append({'path': delete_path, 'method': DELETE})
    if cmd.get('address_family', None) is not None:
        if cmd['address_family'].get('afis', None) is None:
            delete_path = delete_static_path + '/afi-safis/afi-safi'
            requests.append({'path': delete_path, 'method': DELETE})
        else:
            for each in cmd['address_family']['afis']:
                afi = each.get('afi', None)
                safi = each.get('safi', None)
                activate = each.get('activate', None)
                allowas_in = each.get('allowas_in', None)
                ip_afi = each.get('ip_afi', None)
                prefix_limit = each.get('prefix_limit', None)
                prefix_list_in = each.get('prefix_list_in', None)
                prefix_list_out = each.get('prefix_list_out', None)
                afi_safi = afi.upper() + '_' + safi.upper()
                afi_safi_name = 'openconfig-bgp-types:' + afi_safi
                if afi and safi and (not activate) and (not allowas_in) and (not ip_afi) and (not prefix_limit) and (not prefix_list_in) and (not prefix_list_out):
                    delete_path = delete_static_path + '/afi-safis/afi-safi=%s' % afi_safi_name
                    requests.append({'path': delete_path, 'method': DELETE})
                else:
                    if activate:
                        delete_path = delete_static_path + '/afi-safis/afi-safi=%s/config/enabled' % afi_safi_name
                        requests.append({'path': delete_path, 'method': DELETE})
                    if allowas_in:
                        if allowas_in.get('origin', None):
                            delete_path = delete_static_path + '/afi-safis/afi-safi=%s/allow-own-as/config/origin' % afi_safi_name
                            requests.append({'path': delete_path, 'method': DELETE})
                        if allowas_in.get('value', None):
                            delete_path = delete_static_path + '/afi-safis/afi-safi=%s/allow-own-as/config/as-count' % afi_safi_name
                            requests.append({'path': delete_path, 'method': DELETE})
                    if prefix_list_in:
                        delete_path = delete_static_path + '/afi-safis/afi-safi=%s/prefix-list/config/import-policy' % afi_safi_name
                        requests.append({'path': delete_path, 'method': DELETE})
                    if prefix_list_out:
                        delete_path = delete_static_path + '/afi-safis/afi-safi=%s/prefix-list/config/export-policy' % afi_safi_name
                        requests.append({'path': delete_path, 'method': DELETE})
                    if afi_safi == 'IPV4_UNICAST':
                        if ip_afi:
                            requests.extend(self.delete_ip_afi_requests(ip_afi, afi_safi_name, 'ipv4-unicast', delete_static_path))
                        if prefix_limit:
                            requests.extend(self.delete_prefix_limit_requests(prefix_limit, afi_safi_name, 'ipv4-unicast', delete_static_path))
                    elif afi_safi == 'IPV6_UNICAST':
                        if ip_afi:
                            requests.extend(self.delete_ip_afi_requests(ip_afi, afi_safi_name, 'ipv6-unicast', delete_static_path))
                        if prefix_limit:
                            requests.extend(self.delete_prefix_limit_requests(prefix_limit, afi_safi_name, 'ipv6-unicast', delete_static_path))
    return requests