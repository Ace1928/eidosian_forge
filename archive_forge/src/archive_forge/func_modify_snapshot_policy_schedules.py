from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_snapshot_policy_schedules(self, current):
    """
        Modify existing schedules in snapshot policy
        :return: None
        """
    self.validate_parameters()
    delete_schedules, modify_schedules, add_schedules = ([], [], [])
    if 'snapmirror_label' in self.parameters:
        snapmirror_labels = self.parameters['snapmirror_label']
    else:
        snapmirror_labels = [None] * len(self.parameters['schedule'])
    for schedule in current['schedule']:
        schedule = self.safe_strip(schedule)
        if schedule not in [item.strip() for item in self.parameters['schedule']]:
            options = {'policy': current['name'], 'schedule': schedule}
            delete_schedules.append(options)
    for schedule, count, snapmirror_label in zip(self.parameters['schedule'], self.parameters['count'], snapmirror_labels):
        schedule = self.safe_strip(schedule)
        snapmirror_label = self.safe_strip(snapmirror_label)
        options = {'policy': current['name'], 'schedule': schedule}
        if schedule in current['schedule']:
            modify = False
            schedule_index = current['schedule'].index(schedule)
            if count != current['count'][schedule_index]:
                options['new-count'] = str(count)
                modify = True
            if snapmirror_label is not None and snapmirror_label != current['snapmirror_label'][schedule_index]:
                options['new-snapmirror-label'] = snapmirror_label
                modify = True
            if modify:
                modify_schedules.append(options)
        else:
            options['count'] = str(count)
            if snapmirror_label is not None and snapmirror_label != '':
                options['snapmirror-label'] = snapmirror_label
            add_schedules.append(options)
    while len(delete_schedules) > 1:
        options = delete_schedules.pop()
        self.modify_snapshot_policy_schedule(options, 'snapshot-policy-remove-schedule')
    while modify_schedules:
        options = modify_schedules.pop()
        self.modify_snapshot_policy_schedule(options, 'snapshot-policy-modify-schedule')
    if add_schedules:
        options = add_schedules.pop()
        self.modify_snapshot_policy_schedule(options, 'snapshot-policy-add-schedule')
    while delete_schedules:
        options = delete_schedules.pop()
        self.modify_snapshot_policy_schedule(options, 'snapshot-policy-remove-schedule')
    while add_schedules:
        options = add_schedules.pop()
        self.modify_snapshot_policy_schedule(options, 'snapshot-policy-add-schedule')