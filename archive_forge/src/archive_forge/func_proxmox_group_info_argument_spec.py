from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.proxmox import (
def proxmox_group_info_argument_spec():
    return dict(group=dict(type='str', aliases=['groupid', 'name']))