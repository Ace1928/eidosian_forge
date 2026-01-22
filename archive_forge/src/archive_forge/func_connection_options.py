from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_text
import re
def connection_options(self, detect_change=False):
    options = {'connection.autoconnect': self.autoconnect, 'connection.zone': self.zone}
    if self.ip_conn_type and (not self.master):
        options.update({'ipv4.addresses': self.enforce_ipv4_cidr_notation(self.ip4), 'ipv4.dhcp-client-id': self.dhcp_client_id, 'ipv4.dns': self.dns4, 'ipv4.dns-search': self.dns4_search, 'ipv4.dns-options': self.dns4_options, 'ipv4.ignore-auto-dns': self.dns4_ignore_auto, 'ipv4.gateway': self.gw4, 'ipv4.ignore-auto-routes': self.gw4_ignore_auto, 'ipv4.routes': self.enforce_routes_format(self.routes4, self.routes4_extended), 'ipv4.route-metric': self.route_metric4, 'ipv4.routing-rules': self.routing_rules4, 'ipv4.never-default': self.never_default4, 'ipv4.method': self.ipv4_method, 'ipv4.may-fail': self.may_fail4, 'ipv6.addresses': self.enforce_ipv6_cidr_notation(self.ip6), 'ipv6.dns': self.dns6, 'ipv6.dns-search': self.dns6_search, 'ipv6.dns-options': self.dns6_options, 'ipv6.ignore-auto-dns': self.dns6_ignore_auto, 'ipv6.gateway': self.gw6, 'ipv6.ignore-auto-routes': self.gw6_ignore_auto, 'ipv6.routes': self.enforce_routes_format(self.routes6, self.routes6_extended), 'ipv6.route-metric': self.route_metric6, 'ipv6.method': self.ipv6_method, 'ipv6.ip6-privacy': self.ip_privacy6, 'ipv6.addr-gen-mode': self.addr_gen_mode6})
        if self.ipv4_method and self.ipv4_method != 'disabled':
            options.update({'ipv4.may-fail': self.may_fail4})
    if self.mac:
        options.update({self.mac_setting: self.mac})
    if self.mtu_conn_type:
        options.update({self.mtu_setting: self.mtu})
    if self.slave_conn_type:
        options.update({'connection.master': self.master, 'connection.slave-type': self.slave_type})
    if self.type == 'bond':
        options.update({'arp-interval': self.arp_interval, 'arp-ip-target': self.arp_ip_target, 'downdelay': self.downdelay, 'miimon': self.miimon, 'mode': self.mode, 'primary': self.primary, 'updelay': self.updelay, 'xmit_hash_policy': self.xmit_hash_policy})
    elif self.type == 'bond-slave':
        if self.slave_type and self.slave_type != 'bond':
            self.module.fail_json(msg="Connection type '%s' cannot be combined with '%s' slave-type. Allowed slave-type for '%s' is 'bond'." % (self.type, self.slave_type, self.type))
        if not self.slave_type:
            self.module.warn("Connection 'slave-type' property automatically set to 'bond' because of using 'bond-slave' connection type.")
            options.update({'connection.slave-type': 'bond'})
    elif self.type == 'bridge':
        options.update({'bridge.ageing-time': self.ageingtime, 'bridge.forward-delay': self.forwarddelay, 'bridge.hello-time': self.hellotime, 'bridge.max-age': self.maxage, 'bridge.priority': self.priority, 'bridge.stp': self.stp})
        if self.stp:
            options.update({'bridge.priority': self.priority})
    elif self.type == 'team':
        options.update({'team.runner': self.runner, 'team.runner-hwaddr-policy': self.runner_hwaddr_policy})
        if self.runner_fast_rate is not None:
            options.update({'team.runner-fast-rate': self.runner_fast_rate})
    elif self.type == 'bridge-slave':
        if self.slave_type and self.slave_type != 'bridge':
            self.module.fail_json(msg="Connection type '%s' cannot be combined with '%s' slave-type. Allowed slave-type for '%s' is 'bridge'." % (self.type, self.slave_type, self.type))
        if not self.slave_type:
            self.module.warn("Connection 'slave-type' property automatically set to 'bridge' because of using 'bridge-slave' connection type.")
            options.update({'connection.slave-type': 'bridge'})
        self.module.warn("Connection type as 'bridge-slave' implies 'ethernet' connection with 'bridge' slave-type. Consider using slave_type='bridge' with necessary type.")
        options.update({'bridge-port.path-cost': self.path_cost, 'bridge-port.hairpin-mode': self.hairpin, 'bridge-port.priority': self.slavepriority})
    elif self.type == 'team-slave':
        if self.slave_type and self.slave_type != 'team':
            self.module.fail_json(msg="Connection type '%s' cannot be combined with '%s' slave-type. Allowed slave-type for '%s' is 'team'." % (self.type, self.slave_type, self.type))
        if not self.slave_type:
            self.module.warn("Connection 'slave-type' property automatically set to 'team' because of using 'team-slave' connection type.")
            options.update({'connection.slave-type': 'team'})
    elif self.tunnel_conn_type:
        options.update({'ip-tunnel.local': self.ip_tunnel_local, 'ip-tunnel.mode': self.type, 'ip-tunnel.parent': self.ip_tunnel_dev, 'ip-tunnel.remote': self.ip_tunnel_remote})
        if self.type == 'gre':
            options.update({'ip-tunnel.input-key': self.ip_tunnel_input_key, 'ip-tunnel.output-key': self.ip_tunnel_output_key})
    elif self.type == 'vlan':
        options.update({'vlan.id': self.vlanid, 'vlan.parent': self.vlandev, 'vlan.flags': self.flags, 'vlan.ingress': self.ingress, 'vlan.egress': self.egress})
    elif self.type == 'vxlan':
        options.update({'vxlan.id': self.vxlan_id, 'vxlan.local': self.vxlan_local, 'vxlan.remote': self.vxlan_remote})
    elif self.type == 'wifi':
        options.update({'802-11-wireless.ssid': self.ssid, 'connection.slave-type': ('bond' if self.slave_type is None else self.slave_type) if self.master else None})
        if self.wifi:
            for name, value in self.wifi.items():
                options.update({'802-11-wireless.%s' % name: value})
        if self.wifi_sec:
            for name, value in self.wifi_sec.items():
                options.update({'802-11-wireless-security.%s' % name: value})
    elif self.type == 'gsm':
        if self.gsm:
            for name, value in self.gsm.items():
                options.update({'gsm.%s' % name: value})
    elif self.type == 'macvlan':
        if self.macvlan:
            for name, value in self.macvlan.items():
                options.update({'macvlan.%s' % name: value})
        elif self.state == 'present':
            raise NmcliModuleError('type is macvlan but all of the following are missing: macvlan')
    elif self.type == 'wireguard':
        if self.wireguard:
            for name, value in self.wireguard.items():
                options.update({'wireguard.%s' % name: value})
    elif self.type == 'vpn':
        if self.vpn:
            vpn_data_values = ''
            for name, value in self.vpn.items():
                if name == 'service-type':
                    options.update({'vpn.service-type': value})
                elif name == 'permissions':
                    options.update({'connection.permissions': value})
                else:
                    if vpn_data_values != '':
                        vpn_data_values += ', '
                    if isinstance(value, bool):
                        value = self.bool_to_string(value)
                    vpn_data_values += '%s=%s' % (name, value)
                options.update({'vpn.data': vpn_data_values})
    elif self.type == 'infiniband':
        options.update({'infiniband.transport-mode': self.transport_mode})
    for setting, value in options.items():
        setting_type = self.settings_type(setting)
        convert_func = None
        if setting_type is bool:
            convert_func = self.bool_to_string
        if detect_change:
            if setting in ('vlan.id', 'vxlan.id'):
                convert_func = to_text
            elif setting == self.mtu_setting:
                convert_func = self.mtu_to_string
            elif setting == 'ipv6.ip6-privacy':
                convert_func = self.ip6_privacy_to_num
        elif setting_type is list:
            convert_func = self.list_to_string
        if callable(convert_func):
            options[setting] = convert_func(options[setting])
    return options