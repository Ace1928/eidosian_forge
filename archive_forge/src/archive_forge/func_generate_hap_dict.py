from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
import time
import http
@_api_permission_denied_handler('host_access_policies')
def generate_hap_dict(module, fusion):
    hap_info = {}
    api_instance = purefusion.HostAccessPoliciesApi(fusion)
    hosts = api_instance.list_host_access_policies()
    for host in hosts.items:
        name = host.name
        hap_info[name] = {'personality': host.personality, 'display_name': host.display_name, 'iqn': host.iqn}
    return hap_info