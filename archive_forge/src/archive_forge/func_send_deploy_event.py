from __future__ import absolute_import, division, print_function
import json
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.urls import fetch_url
def send_deploy_event(module, key, revision_id, deployed_by='Ansible', deployed_to=None, repository=None):
    """Send a deploy event to Stackdriver"""
    deploy_api = 'https://event-gateway.stackdriver.com/v1/deployevent'
    params = {}
    params['revision_id'] = revision_id
    params['deployed_by'] = deployed_by
    if deployed_to:
        params['deployed_to'] = deployed_to
    if repository:
        params['repository'] = repository
    return do_send_request(module, deploy_api, params, key)