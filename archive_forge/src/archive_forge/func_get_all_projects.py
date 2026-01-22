from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.urls import fetch_url
def get_all_projects(self):
    """Fetches all projects."""
    self.projects = self.rest.get_paginated_data(base_url='projects?', data_key_name='projects')