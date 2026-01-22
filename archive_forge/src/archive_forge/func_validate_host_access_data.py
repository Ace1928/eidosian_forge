from __future__ import absolute_import, division, print_function
import re
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def validate_host_access_data(self, host_dict):
    """
        Validate host access data
        :param host_dict: Host access data
        :return None
        """
    fqdn_pat = re.compile('(?=^.{4,253}$)(^((?!-)[a-zA-Z0-9-]{0,62}[a-zA-Z0-9]\\.)+[a-zA-Z]{2,63}$)')
    if host_dict.get('host_name'):
        version = get_ip_version(host_dict.get('host_name'))
        if version in (4, 6):
            msg = 'IP4/IP6: %s given in host_name instead of name' % host_dict.get('host_name')
            LOG.error(msg)
            self.module.fail_json(msg=msg)
    if host_dict.get('ip_address'):
        ip_or_fqdn = host_dict.get('ip_address')
        version = get_ip_version(ip_or_fqdn)
        if version == 0 and (not fqdn_pat.match(ip_or_fqdn)):
            msg = '%s is not a valid FQDN' % ip_or_fqdn
            LOG.error(msg)
            self.module.fail_json(msg=msg)
    if host_dict.get('subnet'):
        subnet = host_dict.get('subnet')
        subnet_info = subnet.split('/')
        if len(subnet_info) != 2:
            msg = "Subnet should be in format 'IP address/netmask' or 'IP address/prefix length'"
            LOG.error(msg)
            self.module.fail_json(msg=msg)