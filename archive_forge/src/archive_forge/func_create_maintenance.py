from __future__ import absolute_import, division, print_function
import datetime
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.urls import open_url
def create_maintenance(auth_headers, url, statuspage, host_ids, all_infrastructure_affected, automation, title, desc, returned_date, maintenance_notify_now, maintenance_notify_72_hr, maintenance_notify_24_hr, maintenance_notify_1_hr):
    component_id = []
    container_id = []
    for val in host_ids:
        component_id.append(val['component_id'])
        container_id.append(val['container_id'])
    infrastructure_id = [i + '-' + j for i, j in zip(component_id, container_id)]
    try:
        values = json.dumps({'statuspage_id': statuspage, 'all_infrastructure_affected': str(int(all_infrastructure_affected)), 'infrastructure_affected': infrastructure_id, 'automation': str(int(automation)), 'maintenance_name': title, 'maintenance_details': desc, 'date_planned_start': returned_date[0], 'time_planned_start': returned_date[1], 'date_planned_end': returned_date[2], 'time_planned_end': returned_date[3], 'maintenance_notify_now': str(int(maintenance_notify_now)), 'maintenance_notify_72_hr': str(int(maintenance_notify_72_hr)), 'maintenance_notify_24_hr': str(int(maintenance_notify_24_hr)), 'maintenance_notify_1_hr': str(int(maintenance_notify_1_hr))})
        response = open_url(url + '/v2/maintenance/schedule', data=values, headers=auth_headers)
        data = json.loads(response.read())
        if data['status']['error'] == 'yes':
            return (1, None, data['status']['message'])
    except Exception as e:
        return (1, None, to_native(e))
    return (0, None, None)