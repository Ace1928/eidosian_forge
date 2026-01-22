from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
def gather_ntp_info(self):
    hosts_info = {}
    for host in self.hosts:
        host_ntp_info = []
        host_date_time_manager = host.configManager.dateTimeSystem
        if host_date_time_manager:
            host_ntp_info.append(dict(time_zone_identifier=host_date_time_manager.dateTimeInfo.timeZone.key, time_zone_name=host_date_time_manager.dateTimeInfo.timeZone.name, time_zone_description=host_date_time_manager.dateTimeInfo.timeZone.description, time_zone_gmt_offset=host_date_time_manager.dateTimeInfo.timeZone.gmtOffset, ntp_servers=list(host_date_time_manager.dateTimeInfo.ntpConfig.server)))
        hosts_info[host.name] = host_ntp_info
    return hosts_info