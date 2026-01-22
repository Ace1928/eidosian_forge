from __future__ import absolute_import, division, print_function
import json
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import request, eseries_host_argument_spec
def has_match(module, ssid, api_url, api_pwd, api_usr, body):
    compare_keys = ['syncIntervalMinutes', 'syncWarnThresholdMinutes', 'recoveryWarnThresholdMinutes', 'repoUtilizationWarnThreshold']
    desired_state = dict(((x, body.get(x)) for x in compare_keys))
    label_exists = False
    matches_spec = False
    current_state = None
    async_id = None
    api_data = None
    desired_name = body.get('name')
    endpoint = 'storage-systems/%s/async-mirrors' % ssid
    url = api_url + endpoint
    try:
        rc, data = request(url, url_username=api_usr, url_password=api_pwd, headers=HEADERS)
    except Exception as e:
        module.exit_json(msg='Error finding a match. Message: %s' % to_native(e), exception=traceback.format_exc())
    for async_group in data:
        if async_group['label'] == desired_name:
            label_exists = True
            api_data = async_group
            async_id = async_group['groupRef']
            current_state = dict(syncIntervalMinutes=async_group['syncIntervalMinutes'], syncWarnThresholdMinutes=async_group['syncCompletionTimeAlertThresholdMinutes'], recoveryWarnThresholdMinutes=async_group['recoveryPointAgeAlertThresholdMinutes'], repoUtilizationWarnThreshold=async_group['repositoryUtilizationWarnThreshold'])
    if current_state == desired_state:
        matches_spec = True
    return (label_exists, matches_spec, api_data, async_id)