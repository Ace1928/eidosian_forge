from __future__ import absolute_import, division, print_function
import json
import os
from time import sleep
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.urls import url_argument_spec
def sync_repo(self, repo_id):
    url = '%s/pulp/api/v2/repositories/%s/actions/sync/' % (self.host, repo_id)
    response, info = fetch_url(self.module, url, data='', method='POST')
    if info['status'] != 202:
        self.module.fail_json(msg='Failed to schedule a sync of the repo.', status_code=info['status'], response=info['msg'], url=url)
    if self.wait_for_completion:
        self.verify_tasks_completed(json.load(response))
    return True