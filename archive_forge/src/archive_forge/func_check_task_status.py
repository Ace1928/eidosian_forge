from __future__ import (absolute_import, division, print_function)
import logging
import time
from ansible.module_utils.basic import missing_required_lib
def check_task_status(self, api_url):
    headers = {'X-Agent-Id': self.format_client_id(self.module.params['client_id'])}
    network_retries = 3
    while True:
        result, error, dummy = self.get(api_url, None, header=headers)
        if error is not None:
            if network_retries <= 0:
                return (0, '', error)
            time.sleep(1)
            network_retries -= 1
        else:
            response = result
            break
    return (response['status'], response['error'], None)