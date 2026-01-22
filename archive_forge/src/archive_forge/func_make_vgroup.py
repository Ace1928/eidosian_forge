from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.common import (
def make_vgroup(module, array):
    """Create Volume Group"""
    changed = True
    api_version = array._list_available_rest_versions()
    if module.params['bw_qos'] or (module.params['iops_qos'] and VG_IOPS_VERSION in api_version):
        if module.params['bw_qos'] and (not module.params['iops_qos']):
            if int(human_to_bytes(module.params['bw_qos'])) in range(1048576, 549755813888):
                changed = True
                if not module.check_mode:
                    try:
                        array.create_vgroup(module.params['name'], bandwidth_limit=module.params['bw_qos'])
                    except Exception:
                        module.fail_json(msg='Vgroup {0} creation failed.'.format(module.params['name']))
            else:
                module.fail_json(msg='Bandwidth QoS value {0} out of range.'.format(module.params['bw_qos']))
        elif module.params['iops_qos'] and (not module.params['bw_qos']):
            if int(human_to_real(module.params['iops_qos'])) in range(100, 100000000):
                changed = True
                if not module.check_mode:
                    try:
                        array.create_vgroup(module.params['name'], iops_limit=module.params['iops_qos'])
                    except Exception:
                        module.fail_json(msg='Vgroup {0} creation failed.'.format(module.params['name']))
            else:
                module.fail_json(msg='IOPs QoS value {0} out of range.'.format(module.params['iops_qos']))
        else:
            bw_qos_size = int(human_to_bytes(module.params['bw_qos']))
            if int(human_to_real(module.params['iops_qos'])) in range(100, 100000000) and bw_qos_size in range(1048576, 549755813888):
                changed = True
                if not module.check_mode:
                    try:
                        array.create_vgroup(module.params['name'], iops_limit=module.params['iops_qos'], bandwidth_limit=module.params['bw_qos'])
                    except Exception:
                        module.fail_json(msg='Vgroup {0} creation failed.'.format(module.params['name']))
            else:
                module.fail_json(msg='IOPs or Bandwidth QoS value out of range.')
    else:
        changed = True
        if not module.check_mode:
            try:
                array.create_vgroup(module.params['name'])
            except Exception:
                module.fail_json(msg='creation of volume group {0} failed.'.format(module.params['name']))
    if PRIORITY_API_VERSION in api_version:
        array = get_array(module)
        volume_group = flasharray.VolumeGroup(priority_adjustment=flasharray.PriorityAdjustment(priority_adjustment_operator=module.params['priority_operator'], priority_adjustment_value=module.params['priority_value']))
        if not module.check_mode:
            res = array.patch_volume_groups(names=[module.params['name']], volume_group=volume_group)
            if res.status_code != 200:
                module.fail_json(msg='Failed to set priority adjustment for volume group {0}. Error: {1}'.format(module.params['name'], res.errors[0].message))
    module.exit_json(changed=changed)