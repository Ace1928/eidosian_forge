from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
@staticmethod
def create_kmip_server_info(kms_info, proxy_user_info):
    kmip_server_info = None
    if kms_info:
        kmip_server_info = vim.encryption.KmipServerInfo()
        kmip_server_info.name = kms_info.get('kms_name')
        kmip_server_info.address = kms_info.get('kms_ip')
        if kms_info.get('kms_port') is None:
            kmip_server_info.port = 5696
        else:
            kmip_server_info.port = kms_info.get('kms_port')
        if proxy_user_info:
            if proxy_user_info.get('proxy_server'):
                kmip_server_info.proxyAddress = proxy_user_info['proxy_server']
            if proxy_user_info.get('proxy_port'):
                kmip_server_info.proxyPort = proxy_user_info['proxy_port']
            if proxy_user_info.get('kms_username'):
                kmip_server_info.userName = proxy_user_info['kms_username']
    return kmip_server_info