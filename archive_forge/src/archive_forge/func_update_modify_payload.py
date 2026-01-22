from __future__ import (absolute_import, division, print_function)
import json
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import strip_substr_dict
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
def update_modify_payload(discovery_modify_payload, current_payload, new_name=None):
    parent_items = ['DiscoveryConfigGroupName', 'TrapDestination', 'CommunityString', 'DiscoveryStatusEmailRecipient', 'CreateGroup', 'UseAllProfiles']
    for key in parent_items:
        if key not in discovery_modify_payload and key in current_payload:
            discovery_modify_payload[key] = current_payload[key]
    if not discovery_modify_payload.get('Schedule'):
        exist_schedule = current_payload.get('Schedule', {})
        schedule_payload = {}
        if exist_schedule.get('Cron') == 'startnow':
            schedule_payload['RunNow'] = True
            schedule_payload['RunLater'] = False
            schedule_payload['Cron'] = 'startnow'
        else:
            schedule_payload['RunNow'] = False
            schedule_payload['RunLater'] = True
            schedule_payload['Cron'] = exist_schedule.get('Cron')
        discovery_modify_payload['Schedule'] = schedule_payload
    discovery_modify_payload['DiscoveryConfigGroupId'] = current_payload['DiscoveryConfigGroupId']
    if new_name:
        discovery_modify_payload['DiscoveryConfigGroupName'] = new_name