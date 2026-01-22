from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def build_downtime(module):
    downtime = Downtime()
    if module.params['monitor_tags']:
        downtime.monitor_tags = module.params['monitor_tags']
    if module.params['scope']:
        downtime.scope = module.params['scope']
    if module.params['monitor_id']:
        downtime.monitor_id = module.params['monitor_id']
    if module.params['downtime_message']:
        downtime.message = module.params['downtime_message']
    if module.params['start']:
        downtime.start = module.params['start']
    if module.params['end']:
        downtime.end = module.params['end']
    if module.params['timezone']:
        downtime.timezone = module.params['timezone']
    if module.params['rrule']:
        downtime.recurrence = DowntimeRecurrence(rrule=module.params['rrule'], type='rrule')
    return downtime