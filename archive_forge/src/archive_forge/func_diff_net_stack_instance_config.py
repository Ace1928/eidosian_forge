from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_text
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
def diff_net_stack_instance_config(self):
    """
        Check the difference between a new and existing config.
        """
    self.change_flag = False
    self.diff_config = dict(before={}, after={})
    for key, value in self.enabled_net_stack_instance.items():
        if value is True:
            self.diff_config['before'][key] = {}
            self.diff_config['after'][key] = {}
    if self.enabled_net_stack_instance['default']:
        exist_dns_servers = self.exist_net_stack_instance_config['default'].dnsConfig.address
        for key in ('before', 'after'):
            self.diff_config[key]['default'] = dict(hostname=self.exist_net_stack_instance_config['default'].dnsConfig.hostName, domain=self.exist_net_stack_instance_config['default'].dnsConfig.domainName, preferred_dns=exist_dns_servers[0] if [dns for dns in exist_dns_servers if exist_dns_servers.index(dns) == 0] else None, alternate_dns=exist_dns_servers[1] if [dns for dns in exist_dns_servers if exist_dns_servers.index(dns) == 1] else None, search_domains=self.exist_net_stack_instance_config['default'].dnsConfig.searchDomain, gateway=self.exist_net_stack_instance_config['default'].ipRouteConfig.defaultGateway, ipv6_gateway=self.exist_net_stack_instance_config['default'].ipRouteConfig.ipV6DefaultGateway, congestion_algorithm=self.exist_net_stack_instance_config['default'].congestionControlAlgorithm, max_num_connections=self.exist_net_stack_instance_config['default'].requestedMaxNumberOfConnections)
            if self.default:
                if self.diff_config['before']['default']['hostname'] != self.default['hostname']:
                    self.change_flag = True
                    self.diff_config['after']['default']['hostname'] = self.default['hostname']
                if self.diff_config['before']['default']['domain'] != self.default['domain']:
                    self.change_flag = True
                    self.diff_config['after']['default']['domain'] = self.default['domain']
                if self.diff_config['before']['default']['preferred_dns'] != self.default['preferred_dns']:
                    self.change_flag = True
                    self.diff_config['after']['default']['preferred_dns'] = self.default['preferred_dns']
                if self.diff_config['before']['default']['alternate_dns'] != self.default['alternate_dns']:
                    self.change_flag = True
                    self.diff_config['after']['default']['alternate_dns'] = self.default['alternate_dns']
                if self.diff_config['before']['default']['search_domains'] != self.default['search_domains']:
                    self.change_flag = True
                    self.diff_config['after']['default']['search_domains'] = self.default['search_domains']
                if self.diff_config['before']['default']['gateway'] != self.default['gateway']:
                    self.change_flag = True
                    self.diff_config['after']['default']['gateway'] = self.default['gateway']
                if self.diff_config['before']['default']['ipv6_gateway'] != self.default['ipv6_gateway']:
                    self.change_flag = True
                    self.diff_config['after']['default']['ipv6_gateway'] = self.default['ipv6_gateway']
                if self.diff_config['before']['default']['congestion_algorithm'] != self.default['congestion_algorithm']:
                    self.change_flag = True
                    self.diff_config['after']['default']['congestion_algorithm'] = self.default['congestion_algorithm']
                if self.diff_config['before']['default']['max_num_connections'] != self.default['max_num_connections']:
                    self.change_flag = True
                    self.diff_config['after']['default']['max_num_connections'] = self.default['max_num_connections']
    if self.enabled_net_stack_instance['provisioning']:
        for key in ('before', 'after'):
            self.diff_config[key]['provisioning'] = dict(gateway=self.exist_net_stack_instance_config['provisioning'].ipRouteConfig.defaultGateway, ipv6_gateway=self.exist_net_stack_instance_config['provisioning'].ipRouteConfig.ipV6DefaultGateway, congestion_algorithm=self.exist_net_stack_instance_config['provisioning'].congestionControlAlgorithm, max_num_connections=self.exist_net_stack_instance_config['provisioning'].requestedMaxNumberOfConnections)
        if self.provisioning:
            if self.diff_config['before']['provisioning']['gateway'] != self.provisioning['gateway']:
                self.change_flag = True
                self.diff_config['after']['provisioning']['gateway'] = self.provisioning['gateway']
            if self.diff_config['before']['provisioning']['ipv6_gateway'] != self.provisioning['ipv6_gateway']:
                self.change_flag = True
                self.diff_config['after']['provisioning']['ipv6_gateway'] = self.provisioning['ipv6_gateway']
            if self.diff_config['before']['provisioning']['max_num_connections'] != self.provisioning['max_num_connections']:
                self.change_flag = True
                self.diff_config['after']['provisioning']['max_num_connections'] = self.provisioning['max_num_connections']
            if self.diff_config['before']['provisioning']['congestion_algorithm'] != self.provisioning['congestion_algorithm']:
                self.change_flag = True
                self.diff_config['after']['provisioning']['congestion_algorithm'] = self.provisioning['congestion_algorithm']
    if self.enabled_net_stack_instance['vmotion']:
        for key in ('before', 'after'):
            self.diff_config[key]['vmotion'] = dict(gateway=self.exist_net_stack_instance_config['vmotion'].ipRouteConfig.defaultGateway, ipv6_gateway=self.exist_net_stack_instance_config['vmotion'].ipRouteConfig.ipV6DefaultGateway, congestion_algorithm=self.exist_net_stack_instance_config['vmotion'].congestionControlAlgorithm, max_num_connections=self.exist_net_stack_instance_config['vmotion'].requestedMaxNumberOfConnections)
        if self.vmotion:
            if self.diff_config['before']['vmotion']['gateway'] != self.vmotion['gateway']:
                self.change_flag = True
                self.diff_config['after']['vmotion']['gateway'] = self.vmotion['gateway']
            if self.diff_config['before']['vmotion']['ipv6_gateway'] != self.vmotion['ipv6_gateway']:
                self.change_flag = True
                self.diff_config['after']['vmotion']['ipv6_gateway'] = self.vmotion['ipv6_gateway']
            if self.diff_config['before']['vmotion']['max_num_connections'] != self.vmotion['max_num_connections']:
                self.change_flag = True
                self.diff_config['after']['vmotion']['max_num_connections'] = self.vmotion['max_num_connections']
            if self.diff_config['before']['vmotion']['congestion_algorithm'] != self.vmotion['congestion_algorithm']:
                self.change_flag = True
                self.diff_config['after']['vmotion']['congestion_algorithm'] = self.vmotion['congestion_algorithm']
    if self.enabled_net_stack_instance['vxlan']:
        for key in ('before', 'after'):
            self.diff_config[key]['vxlan'] = dict(gateway=self.exist_net_stack_instance_config['vxlan'].ipRouteConfig.defaultGateway, ipv6_gateway=self.exist_net_stack_instance_config['vxlan'].ipRouteConfig.ipV6DefaultGateway, congestion_algorithm=self.exist_net_stack_instance_config['vxlan'].congestionControlAlgorithm, max_num_connections=self.exist_net_stack_instance_config['vxlan'].requestedMaxNumberOfConnections)
        if self.vxlan:
            if self.diff_config['before']['vxlan']['gateway'] != self.vxlan['gateway']:
                self.change_flag = True
                self.diff_config['after']['vxlan']['gateway'] = self.vxlan['gateway']
            if self.diff_config['before']['vxlan']['ipv6_gateway'] != self.vxlan['ipv6_gateway']:
                self.change_flag = True
                self.diff_config['after']['vxlan']['ipv6_gateway'] = self.vxlan['ipv6_gateway']
            if self.diff_config['before']['vxlan']['max_num_connections'] != self.vxlan['max_num_connections']:
                self.change_flag = True
                self.diff_config['after']['vxlan']['max_num_connections'] = self.vxlan['max_num_connections']
            if self.diff_config['before']['vxlan']['congestion_algorithm'] != self.vxlan['congestion_algorithm']:
                self.change_flag = True
                self.diff_config['after']['vxlan']['congestion_algorithm'] = self.vxlan['congestion_algorithm']