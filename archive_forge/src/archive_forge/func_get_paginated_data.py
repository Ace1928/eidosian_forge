from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.urls import fetch_url
def get_paginated_data(self, base_url=None, data_key_name=None, data_per_page=40, expected_status_code=200):
    """
        Function to get all paginated data from given URL
        Args:
            base_url: Base URL to get data from
            data_key_name: Name of data key value
            data_per_page: Number results per page (Default: 40)
            expected_status_code: Expected returned code from DigitalOcean (Default: 200)
        Returns: List of data

        """
    page = 1
    has_next = True
    ret_data = []
    status_code = None
    response = None
    while has_next or status_code != expected_status_code:
        required_url = '{0}page={1}&per_page={2}'.format(base_url, page, data_per_page)
        response = self.get(required_url)
        status_code = response.status_code
        if status_code != expected_status_code:
            break
        page += 1
        ret_data.extend(response.json[data_key_name])
        try:
            has_next = 'pages' in response.json['links'] and 'next' in response.json['links']['pages']
        except KeyError:
            has_next = False
    if status_code != expected_status_code:
        msg = 'Failed to fetch %s from %s' % (data_key_name, base_url)
        if response:
            msg += ' due to error : %s' % response.json['message']
        self.module.fail_json(msg=msg)
    return ret_data