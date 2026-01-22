from __future__ import absolute_import, division, print_function
from ansible_collections.theforeman.foreman.plugins.module_utils.foreman_helper import ForemanTaxonomicEntityAnsibleModule
def get_provider_info(provider):
    provider_name = provider.lower()
    if provider_name == 'libvirt':
        return ('Libvirt', ['url', 'display_type', 'set_console_password'])
    elif provider_name == 'ovirt':
        return ('Ovirt', ['url', 'user', 'password', 'datacenter', 'use_v4', 'ovirt_quota', 'keyboard_layout', 'public_key'])
    elif provider_name == 'proxmox':
        return ('Proxmox', ['url', 'user', 'password', 'ssl_verify_peer'])
    elif provider_name == 'vmware':
        return ('Vmware', ['url', 'user', 'password', 'datacenter', 'caching_enabled', 'set_console_password'])
    elif provider_name == 'ec2':
        return ('EC2', ['user', 'password', 'region'])
    elif provider_name == 'azurerm':
        return ('AzureRm', ['user', 'password', 'tenant', 'region', 'app_ident', 'cloud', 'sub_id'])
    elif provider_name == 'gce':
        return ('GCE', ['project', 'email', 'key_path', 'zone'])
    elif provider_name == 'openstack':
        return ('Openstack', ['url', 'user', 'password', 'tenant', 'domain', 'project_domain_name', 'project_domain_id'])
    else:
        return ('', [])