from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def get_host_connect_spec(self):
    """
        Function to return Host connection specification
        Returns: host connection specification
        """
    if self.fetch_ssl_thumbprint and self.esxi_ssl_thumbprint == '':
        sslThumbprint = self.get_cert_fingerprint(self.esxi_hostname, self.module.params['port'], self.module.params['proxy_host'], self.module.params['proxy_port'])
    else:
        sslThumbprint = self.esxi_ssl_thumbprint
    host_connect_spec = vim.host.ConnectSpec()
    host_connect_spec.sslThumbprint = sslThumbprint
    host_connect_spec.hostName = self.esxi_hostname
    host_connect_spec.userName = self.esxi_username
    host_connect_spec.password = self.esxi_password
    host_connect_spec.force = self.force_connection
    return host_connect_spec