from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.common import (
def update_vgroup(module, array):
    """Update Volume Group"""
    changed = False
    api_version = array._list_available_rest_versions()
    if PRIORITY_API_VERSION in api_version:
        arrayv6 = get_array(module)
        vg_prio = list(arrayv6.get_volume_groups(names=[module.params['name']]).items)[0].priority_adjustment
        if module.params['priority_operator'] and vg_prio.priority_adjustment_operator != module.params['priority_operator']:
            changed = True
            new_operator = module.params['priority_operator']
        else:
            new_operator = vg_prio.priority_adjustment_operator
        if vg_prio.priority_adjustment_value != module.params['priority_value']:
            changed = True
            new_value = module.params['priority_value']
        else:
            new_value = vg_prio.priority_adjustment_value
        if changed and (not module.check_mode):
            volume_group = flasharray.VolumeGroup(priority_adjustment=flasharray.PriorityAdjustment(priority_adjustment_operator=new_operator, priority_adjustment_value=new_value))
            res = arrayv6.patch_volume_groups(names=[module.params['name']], volume_group=volume_group)
            if res.status_code != 200:
                module.fail_json(msg='Failed to changfe DMM Priority for volume group {0}. Error: {1}'.format(module.params['name'], res.errors[0].message))
    if VG_IOPS_VERSION in api_version:
        try:
            vg_qos = array.get_vgroup(module.params['name'], qos=True)
        except Exception:
            module.fail_json(msg='Failed to get QoS settings for vgroup {0}.'.format(module.params['name']))
    if VG_IOPS_VERSION in api_version:
        if vg_qos['bandwidth_limit'] is None:
            vg_qos['bandwidth_limit'] = 0
        if vg_qos['iops_limit'] is None:
            vg_qos['iops_limit'] = 0
    if module.params['bw_qos'] and VG_IOPS_VERSION in api_version:
        if human_to_bytes(module.params['bw_qos']) != vg_qos['bandwidth_limit']:
            if module.params['bw_qos'] == '0':
                changed = True
                if not module.check_mode:
                    try:
                        array.set_vgroup(module.params['name'], bandwidth_limit='')
                    except Exception:
                        module.fail_json(msg='Vgroup {0} Bandwidth QoS removal failed.'.format(module.params['name']))
            elif int(human_to_bytes(module.params['bw_qos'])) in range(1048576, 549755813888):
                changed = True
                if not module.check_mode:
                    try:
                        array.set_vgroup(module.params['name'], bandwidth_limit=module.params['bw_qos'])
                    except Exception:
                        module.fail_json(msg='Vgroup {0} Bandwidth QoS change failed.'.format(module.params['name']))
            else:
                module.fail_json(msg='Bandwidth QoS value {0} out of range.'.format(module.params['bw_qos']))
    if module.params['iops_qos'] and VG_IOPS_VERSION in api_version:
        if human_to_real(module.params['iops_qos']) != vg_qos['iops_limit']:
            if module.params['iops_qos'] == '0':
                changed = True
                if not module.check_mode:
                    try:
                        array.set_vgroup(module.params['name'], iops_limit='')
                    except Exception:
                        module.fail_json(msg='Vgroup {0} IOPs QoS removal failed.'.format(module.params['name']))
            elif int(human_to_real(module.params['iops_qos'])) in range(100, 100000000):
                changed = True
                if not module.check_mode:
                    try:
                        array.set_vgroup(module.params['name'], iops_limit=module.params['iops_qos'])
                    except Exception:
                        module.fail_json(msg='Vgroup {0} IOPs QoS change failed.'.format(module.params['name']))
            else:
                module.fail_json(msg='Bandwidth QoS value {0} out of range.'.format(module.params['bw_qos']))
    module.exit_json(changed=changed)