from __future__ import absolute_import, division, print_function
import json
import os
from time import sleep
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.urls import url_argument_spec
def compare_repo_distributor_config(self, repo_id, **kwargs):
    repo_config = self.get_repo_config_by_id(repo_id)
    for distributor in repo_config['distributors']:
        for key, value in kwargs.items():
            if key not in distributor['config'].keys():
                return False
            if not distributor['config'][key] == value:
                return False
    return True