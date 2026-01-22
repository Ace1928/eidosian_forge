from __future__ import (absolute_import, division, print_function)
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.dict_transformations import dict_merge
from ansible_collections.community.general.plugins.module_utils.opennebula import flatten, render
def create_exact_count_of_vms(module, client, template_id, exact_count, attributes_dict, count_attributes_dict, labels_list, count_labels_list, disk_size, network_attrs_list, hard, wait, wait_timeout, vm_start_on_hold, vm_persistent, updateconf_dict):
    vm_list = get_all_vms_by_attributes(client, count_attributes_dict, count_labels_list)
    vm_count_diff = exact_count - len(vm_list)
    changed = vm_count_diff != 0
    new_vms_list = []
    instances_list = []
    tagged_instances_list = vm_list
    if module.check_mode:
        return (changed, instances_list, tagged_instances_list)
    if vm_count_diff > 0:
        changed, instances_list, tagged_instances = create_count_of_vms(module, client, template_id, vm_count_diff, attributes_dict, labels_list, disk_size, network_attrs_list, wait, wait_timeout, vm_start_on_hold, vm_persistent, updateconf_dict)
        tagged_instances_list += instances_list
    elif vm_count_diff < 0:
        old_vms_list = []
        while vm_count_diff < 0:
            old_vm = vm_list.pop(0)
            old_vms_list.append(old_vm)
            terminate_vm(module, client, old_vm, hard)
            vm_count_diff += 1
        if wait:
            for vm in old_vms_list:
                wait_for_done(module, client, vm, wait_timeout)
        instances_list = old_vms_list
        old_vms_set = set(old_vms_list)
        tagged_instances_list = [vm for vm in vm_list if vm not in old_vms_set]
    return (changed, instances_list, tagged_instances_list)