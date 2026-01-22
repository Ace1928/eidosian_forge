from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_snapshot_schedule_rest(self, current):
    """
        get details of the snapshot schedule with rest API.
        """
    query = {'snapshot_policy.name': current['name']}
    api = 'storage/snapshot-policies/%s/schedules' % current['uuid']
    fields = 'schedule.name,schedule.uuid,snapmirror_label,count,prefix'
    records, error = rest_generic.get_0_or_more_records(self.rest_api, api, query, fields)
    if error:
        self.module.fail_json(msg='Error on fetching snapshot schedule: %s' % error)
    if records:
        scheduleRecords = {'counts': [], 'prefixes': [], 'schedule_names': [], 'schedule_uuids': [], 'snapmirror_labels': []}
        for item in records:
            scheduleRecords['counts'].append(item['count'])
            scheduleRecords['prefixes'].append(item['prefix'])
            scheduleRecords['schedule_names'].append(item['schedule']['name'])
            scheduleRecords['schedule_uuids'].append(item['schedule']['uuid'])
            scheduleRecords['snapmirror_labels'].append(item['snapmirror_label'])
        return scheduleRecords
    return None