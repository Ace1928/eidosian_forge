from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.common import (
def make_multi_vgroups(module, array):
    """Create multiple Volume Groups"""
    changed = True
    bw_qos_size = iops_qos_size = 0
    names = []
    api_version = array._list_available_rest_versions()
    array = get_array(module)
    for vg_num in range(module.params['start'], module.params['count'] + module.params['start']):
        names.append(module.params['name'] + str(vg_num).zfill(module.params['digits']) + module.params['suffix'])
    if module.params['bw_qos']:
        bw_qos = int(human_to_bytes(module.params['bw_qos']))
        if bw_qos in range(1048576, 549755813888):
            bw_qos_size = bw_qos
        else:
            module.fail_json(msg='Bandwidth QoS value out of range.')
    if module.params['iops_qos']:
        iops_qos = int(human_to_real(module.params['iops_qos']))
        if iops_qos in range(100, 100000000):
            iops_qos_size = iops_qos
        else:
            module.fail_json(msg='IOPs QoS value out of range.')
    if bw_qos_size != 0 and iops_qos_size != 0:
        volume_group = flasharray.VolumeGroupPost(qos=flasharray.Qos(bandwidth_limit=bw_qos_size, iops_limit=iops_qos_size))
    elif bw_qos_size == 0 and iops_qos_size == 0:
        volume_group = flasharray.VolumeGroupPost()
    elif bw_qos_size == 0 and iops_qos_size != 0:
        volume_group = flasharray.VolumeGroupPost(qos=flasharray.Qos(iops_limit=iops_qos_size))
    elif bw_qos_size != 0 and iops_qos_size == 0:
        volume_group = flasharray.VolumeGroupPost(qos=flasharray.Qos(bandwidth_limit=bw_qos_size))
    if not module.check_mode:
        res = array.post_volume_groups(names=names, volume_group=volume_group)
        if res.status_code != 200:
            module.fail_json(msg='Multi-Vgroup {0}#{1} creation failed: {2}'.format(module.params['name'], module.params['suffix'], res.errors[0].message))
        if PRIORITY_API_VERSION in api_version:
            volume_group = flasharray.VolumeGroup(priority_adjustment=flasharray.PriorityAdjustment(priority_adjustment_operator=module.params['priority_operator'], priority_adjustment_value=module.params['priority_value']))
            res = array.patch_volume_groups(names=names, volume_group=volume_group)
            if res.status_code != 200:
                module.fail_json(msg='Failed to set priority adjustments for multi-vgroup {0}#{1}. Error: {2}'.format(module.params['name'], module.params['suffix'], res.errors[0].message))
    module.exit_json(changed=changed)