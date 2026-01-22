from __future__ import absolute_import, division, print_function
import json
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import request, eseries_host_argument_spec
def update_async(module, ssid, api_url, pwd, user, body, new_name, async_id):
    endpoint = 'storage-systems/%s/async-mirrors/%s' % (ssid, async_id)
    url = api_url + endpoint
    compare_keys = ['syncIntervalMinutes', 'syncWarnThresholdMinutes', 'recoveryWarnThresholdMinutes', 'repoUtilizationWarnThreshold']
    desired_state = dict(((x, body.get(x)) for x in compare_keys))
    if new_name:
        desired_state['new_name'] = new_name
    post_data = json.dumps(desired_state)
    try:
        rc, data = request(url, data=post_data, method='POST', headers=HEADERS, url_username=user, url_password=pwd)
    except Exception as e:
        module.exit_json(msg='Exception while updating async mirror group. Message: %s' % to_native(e), exception=traceback.format_exc())
    return data