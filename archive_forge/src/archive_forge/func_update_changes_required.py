from __future__ import absolute_import, division, print_function
from datetime import datetime
import re
from time import sleep
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
def update_changes_required(self):
    """Determine the required state changes for updating an existing consistency group."""
    group = self.get_consistency_group()
    changes = {'update_group': {}, 'add_volumes': [], 'remove_volumes': [], 'expand_reserve_capacity': [], 'trim_reserve_capacity': []}
    if group['alert_threshold_pct'] != self.alert_threshold_pct:
        changes['update_group'].update({'alert_threshold_pct': self.alert_threshold_pct})
    if group['maximum_snapshots'] != self.maximum_snapshots:
        changes['update_group'].update({'maximum_snapshots': self.maximum_snapshots})
    if group['rollback_priority'] != self.rollback_priority:
        changes['update_group'].update({'rollback_priority': self.rollback_priority})
    if group['reserve_capacity_full_policy'] != self.reserve_capacity_full_policy:
        changes['update_group'].update({'reserve_capacity_full_policy': self.reserve_capacity_full_policy})
    remaining_base_volumes = dict(((base_volumes['name'], base_volumes) for base_volumes in group['base_volumes']))
    add_volumes = {}
    expand_volumes = {}
    for volume_name, volume_info in self.volumes.items():
        reserve_capacity_pct = volume_info['reserve_capacity_pct']
        if volume_name in remaining_base_volumes:
            base_volume_reserve_capacity_pct = remaining_base_volumes[volume_name]['reserve_capacity_pct']
            if reserve_capacity_pct > base_volume_reserve_capacity_pct:
                expand_reserve_capacity_pct = reserve_capacity_pct - base_volume_reserve_capacity_pct
                expand_volumes.update({volume_name: {'reserve_capacity_pct': expand_reserve_capacity_pct, 'preferred_reserve_storage_pool': volume_info['preferred_reserve_storage_pool'], 'reserve_volume_id': remaining_base_volumes[volume_name]['repository_volume_info']['id']}})
            elif reserve_capacity_pct < base_volume_reserve_capacity_pct:
                existing_volumes_by_id = self.get_all_volumes_by_id()
                existing_volumes_by_name = self.get_all_volumes_by_name()
                existing_concat_volumes_by_id = self.get_all_concat_volumes_by_id()
                trim_pct = base_volume_reserve_capacity_pct - reserve_capacity_pct
                for timestamp, image in self.get_pit_images_by_timestamp():
                    if existing_volumes_by_id(image['base_volume_id'])['name'] == volume_name:
                        self.module.fail_json(msg='Reserve capacity cannot be trimmed when snapshot images exist for base volume! Base volume [%s]. Group [%s]. Array [%s].' % (volume_name, self.group_name, self.ssid))
                concat_volume_id = remaining_base_volumes[volume_name]['repository_volume_info']['id']
                concat_volume_info = existing_concat_volumes_by_id[concat_volume_id]
                base_volume_info = existing_volumes_by_name[volume_name]
                base_volume_size_bytes = int(base_volume_info['totalSizeInBytes'])
                total_member_volume_size_bytes = 0
                member_volumes_to_trim = []
                for trim_count, member_volume_id in enumerate(reversed(concat_volume_info['memberRefs'][1:])):
                    member_volume_info = existing_volumes_by_id[member_volume_id]
                    member_volumes_to_trim.append(member_volume_info)
                    total_member_volume_size_bytes += int(member_volume_info['totalSizeInBytes'])
                    total_trimmed_size_pct = round(total_member_volume_size_bytes / base_volume_size_bytes * 100)
                    if total_trimmed_size_pct >= trim_pct:
                        changes['trim_reserve_capacity'].append({'concat_volume_id': concat_volume_id, 'trim_count': trim_count + 1})
                        if total_trimmed_size_pct > trim_pct:
                            expand_reserve_capacity_pct = total_trimmed_size_pct - trim_pct
                            expand_volumes.update({volume_name: {'reserve_capacity_pct': expand_reserve_capacity_pct, 'preferred_reserve_storage_pool': volume_info['preferred_reserve_storage_pool'], 'reserve_volume_id': remaining_base_volumes[volume_name]['repository_volume_info']['id']}})
                        break
                else:
                    initial_reserve_volume_info = existing_volumes_by_id[concat_volume_info['memberRefs'][0]]
                    minimum_capacity_pct = round(int(initial_reserve_volume_info['totalSizeInBytes']) / base_volume_size_bytes * 100)
                    self.module.fail_json(msg='Cannot delete initial reserve capacity volume! Minimum reserve capacity percent [%s]. Base volume [%s]. Group [%s]. Array [%s].' % (minimum_capacity_pct, volume_name, self.group_name, self.ssid))
            remaining_base_volumes.pop(volume_name)
        else:
            add_volumes.update({volume_name: {'reserve_capacity_pct': reserve_capacity_pct, 'preferred_reserve_storage_pool': volume_info['preferred_reserve_storage_pool']}})
    changes['add_volumes'] = add_volumes
    changes['expand_reserve_capacity'] = expand_volumes
    changes['remove_volumes'] = remaining_base_volumes
    return changes