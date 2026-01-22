from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.proxmox import (
def proxmox_task_info_argument_spec():
    return dict(task=dict(type='str', aliases=['upid', 'name'], required=False), node=dict(type='str', required=True))