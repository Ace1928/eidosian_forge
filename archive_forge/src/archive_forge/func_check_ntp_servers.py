from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
from ansible.module_utils._text import to_native
def check_ntp_servers(self, host):
    """Check configured NTP servers"""
    update_ntp_list = []
    host_datetime_system = host.configManager.dateTimeSystem
    if host_datetime_system:
        ntp_servers_configured = host_datetime_system.dateTimeInfo.ntpConfig.server
        if self.desired_state:
            for ntp_server in self.ntp_servers:
                if self.desired_state == 'present' and ntp_server not in ntp_servers_configured:
                    update_ntp_list.append(ntp_server)
                if self.desired_state == 'absent' and ntp_server in ntp_servers_configured:
                    update_ntp_list.append(ntp_server)
        elif ntp_servers_configured != self.ntp_servers:
            for ntp_server in self.ntp_servers:
                update_ntp_list.append(ntp_server)
        if update_ntp_list:
            self.results[host.name]['ntp_servers_previous'] = ntp_servers_configured
    return (ntp_servers_configured, update_ntp_list)