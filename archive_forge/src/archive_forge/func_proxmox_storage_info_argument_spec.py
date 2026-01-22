from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.proxmox import (
def proxmox_storage_info_argument_spec():
    return dict(storage=dict(type='str', required=True, aliases=['name']), content=dict(type='str', required=False, default='all', choices=['all', 'backup', 'rootdir', 'images', 'iso']), vmid=dict(type='int'), node=dict(required=True, type='str'))