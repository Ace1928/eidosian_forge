from __future__ import absolute_import, division, print_function
import json
import os
from time import sleep
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.urls import url_argument_spec
def update_repo_importer_config(self, repo_id, **kwargs):
    url = '%s/pulp/api/v2/repositories/%s/importers/' % (self.host, repo_id)
    data = dict()
    importer_config = dict()
    for key, value in kwargs.items():
        if value is not None:
            importer_config[key] = value
    data['importer_config'] = importer_config
    if self.repo_type == 'rpm':
        data['importer_type_id'] = 'yum_importer'
    response, info = fetch_url(self.module, url, data=json.dumps(data), method='POST')
    if info['status'] != 202:
        self.module.fail_json(msg='Failed to set the repo importer configuration', status_code=info['status'], response=info['msg'], importer_config=importer_config, url=url)