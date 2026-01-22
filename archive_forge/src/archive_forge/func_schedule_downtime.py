from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def schedule_downtime(module, api_client):
    downtime = _get_downtime(module, api_client)
    if downtime is None:
        _post_downtime(module, api_client)
    else:
        _update_downtime(module, downtime, api_client)