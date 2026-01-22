from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib, env_fallback
def getInstanceDetails(api, server):
    """
    Return the details of an instance, populating IPs, etc.
    """
    instance = {'id': server['LINODEID'], 'name': server['LABEL'], 'public': [], 'private': []}
    for ip in api.linode_ip_list(LinodeId=server['LINODEID']):
        if ip['ISPUBLIC'] and 'ipv4' not in instance:
            instance['ipv4'] = ip['IPADDRESS']
            instance['fqdn'] = ip['RDNS_NAME']
        if ip['ISPUBLIC']:
            instance['public'].append({'ipv4': ip['IPADDRESS'], 'fqdn': ip['RDNS_NAME'], 'ip_id': ip['IPADDRESSID']})
        else:
            instance['private'].append({'ipv4': ip['IPADDRESS'], 'fqdn': ip['RDNS_NAME'], 'ip_id': ip['IPADDRESSID']})
    return instance