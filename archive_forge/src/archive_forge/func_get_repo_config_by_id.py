from __future__ import absolute_import, division, print_function
import json
import os
from time import sleep
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.urls import url_argument_spec
def get_repo_config_by_id(self, repo_id):
    if repo_id not in self.repo_cache.keys():
        repo_array = [x for x in self.repo_list if x['id'] == repo_id]
        self.repo_cache[repo_id] = repo_array[0]
    return self.repo_cache[repo_id]