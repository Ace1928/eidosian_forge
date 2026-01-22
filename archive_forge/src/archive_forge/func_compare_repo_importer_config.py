from __future__ import absolute_import, division, print_function
import json
import os
from time import sleep
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.urls import url_argument_spec
def compare_repo_importer_config(self, repo_id, **kwargs):
    repo_config = self.get_repo_config_by_id(repo_id)
    for importer in repo_config['importers']:
        for key, value in kwargs.items():
            if value is not None:
                if key not in importer['config'].keys():
                    return False
                if not importer['config'][key] == value:
                    return False
    return True