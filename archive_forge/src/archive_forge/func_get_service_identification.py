from __future__ import absolute_import, division, print_function
import json
import os
import random
import string
import gzip
from io import BytesIO
from ansible.module_utils.urls import open_url
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.six import text_type
from ansible.module_utils.six.moves import http_client
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.module_utils.ansible_release import __version__ as ansible_version
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def get_service_identification(self, manager):
    result = {}
    if manager is None:
        if len(self.manager_uris) == 1:
            manager = self.manager_uris[0].split('/')[-1]
        elif len(self.manager_uris) > 1:
            entries = self.get_multi_manager_inventory()['entries']
            managers = [m[0]['manager_uri'] for m in entries if m[1].get('ServiceIdentification')]
            if len(managers) == 1:
                manager = managers[0].split('/')[-1]
            else:
                self.module.fail_json(msg=['Multiple managers with ServiceIdentification were found: %s' % str(managers), "Please specify by using the 'manager' parameter in your playbook"])
        elif len(self.manager_uris) == 0:
            self.module.fail_json(msg='No manager identities were found')
    response = self.get_request(self.root_uri + '/redfish/v1/Managers/' + manager, override_headers=None)
    try:
        result['service_identification'] = response['data']['ServiceIdentification']
    except Exception as e:
        self.module.fail_json(msg='Service ID not found for manager %s' % manager)
    result['ret'] = True
    return result