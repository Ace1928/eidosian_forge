from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible.module_utils._text import to_native
def create_enabled_services_string(self):
    """Create services list"""
    services = []
    if self.enable_mgmt:
        services.append('Mgmt')
    if self.enable_vmotion:
        services.append('vMotion')
    if self.enable_ft:
        services.append('FT')
    if self.enable_vsan:
        services.append('VSAN')
    if self.enable_provisioning:
        services.append('Prov')
    if self.enable_replication:
        services.append('Repl')
    if self.enable_replication_nfc:
        services.append('Repl_NFC')
    if self.enable_backup_nfc:
        services.append('Backup_NFC')
    return ', '.join(services)