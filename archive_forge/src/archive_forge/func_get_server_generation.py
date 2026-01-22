from __future__ import (absolute_import, division, print_function)
import json
import re
import time
import os
from ansible.module_utils.urls import open_url, ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.common.parameters import env_fallback
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import config_ipv6
@property
def get_server_generation(self):
    """
        This method fetches the connected server generation.
        :return: 14, 4.11.11.11
        """
    firmware_version = None
    response = self.invoke_request(MANAGER_URI, 'GET')
    if response.status_code == 200:
        generation = int(re.search('\\d+(?=G)', response.json_data['Model']).group())
        firmware_version = response.json_data['FirmwareVersion']
    return (generation, firmware_version)