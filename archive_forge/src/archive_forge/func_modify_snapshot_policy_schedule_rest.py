from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_snapshot_policy_schedule_rest(self, modify, current):
    """
        Modify snapshot schedule with rest API.
        """
    if not self.use_rest:
        return self.modify_snapshot_policy_schedules(current)
    schedule_info = None
    api = 'storage/snapshot-policies/%s/schedules' % current['uuid']
    schedule_info = self.get_snapshot_schedule_rest(current)
    delete_schedules, modify_schedules, add_schedules = ([], [], [])
    retain_schedules_count = 0
    if 'snapmirror_label' in self.parameters:
        snapmirror_labels = self.parameters['snapmirror_label']
    else:
        snapmirror_labels = [None] * len(self.parameters['schedule'])
    if 'prefix' in self.parameters:
        prefixes = self.parameters['prefix']
    else:
        prefixes = [None] * len(self.parameters['schedule'])
    for schedule_name, schedule_uuid in zip(schedule_info['schedule_names'], schedule_info['schedule_uuids']):
        schedule_name = self.safe_strip(schedule_name)
        if schedule_name not in [item.strip() for item in self.parameters['schedule']]:
            delete_schedules.append(schedule_uuid)
        else:
            retain_schedules_count += 1
    for schedule_name, count, snapmirror_label, prefix in zip(self.parameters['schedule'], self.parameters['count'], snapmirror_labels, prefixes):
        schedule_name = self.safe_strip(schedule_name)
        if snapmirror_label:
            snapmirror_label = self.safe_strip(snapmirror_label)
        if prefix:
            prefix = self.safe_strip(prefix)
        body = {}
        if schedule_name in schedule_info['schedule_names']:
            modify = False
            schedule_index = schedule_info['schedule_names'].index(schedule_name)
            schedule_uuid = schedule_info['schedule_uuids'][schedule_index]
            if count != schedule_info['counts'][schedule_index]:
                body['count'] = str(count)
                modify = True
            if snapmirror_label is not None and snapmirror_label != schedule_info['snapmirror_labels'][schedule_index]:
                body['snapmirror_label'] = snapmirror_label
                modify = True
            if prefix is not None and prefix != schedule_info['prefixes'][schedule_index]:
                body['prefix'] = prefix
                modify = True
            if modify:
                body['schedule_uuid'] = schedule_uuid
                modify_schedules.append(body)
        else:
            body['schedule.name'] = schedule_name
            body['count'] = str(count)
            if snapmirror_label is not None and snapmirror_label != '':
                body['snapmirror_label'] = snapmirror_label
            if prefix is not None and prefix != '':
                body['prefix'] = prefix
            add_schedules.append(body)
    count = 0 if retain_schedules_count > 0 else 1
    while len(delete_schedules) > count:
        schedule_uuid = delete_schedules.pop()
        record, error = rest_generic.delete_async(self.rest_api, api, schedule_uuid)
        if error is not None:
            self.module.fail_json(msg='Error on deleting snapshot policy schedule: %s' % error)
    while modify_schedules:
        body = modify_schedules.pop()
        schedule_id = body.pop('schedule_uuid')
        record, error = rest_generic.patch_async(self.rest_api, api, schedule_id, body)
        if error is not None:
            self.module.fail_json(msg='Error on modifying snapshot policy schedule: %s' % error)
    if add_schedules:
        body = add_schedules.pop()
        record, error = rest_generic.post_async(self.rest_api, api, body)
        if error is not None:
            self.module.fail_json(msg='Error on adding snapshot policy schedule: %s' % error)
    while delete_schedules:
        schedule_uuid = delete_schedules.pop()
        record, error = rest_generic.delete_async(self.rest_api, api, schedule_uuid)
        if error is not None:
            self.module.fail_json(msg='Error on deleting snapshot policy schedule: %s' % error)
    while add_schedules:
        body = add_schedules.pop()
        record, error = rest_generic.post_async(self.rest_api, api, body)
        if error is not None:
            self.module.fail_json(msg='Error on adding snapshot policy schedule: %s' % error)