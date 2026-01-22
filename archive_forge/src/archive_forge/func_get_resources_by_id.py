from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.urls import fetch_url
def get_resources_by_id(self, id):
    """Fetches the project resources with the given id.

        Returns:
        error_message -- project fetch error message (or "" if no error)
        resources -- resources dictionary representation (or {} if error)
        """
    resources = self.rest.get_paginated_data(base_url='projects/{0}/resources?'.format(id), data_key_name='resources')
    return ('', dict(resources=resources))