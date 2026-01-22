from __future__ import absolute_import, division, print_function
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell import (
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.powerflex_base \
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.configuration \
from ansible.module_utils.basic import AnsibleModule
def get_powerflex_fault_set_parameters():
    """This method provide parameter required for the Ansible Fault Set module on
    PowerFlex"""
    return dict(fault_set_name=dict(), fault_set_id=dict(), protection_domain_name=dict(), protection_domain_id=dict(), fault_set_new_name=dict(), state=dict(default='present', choices=['present', 'absent']))