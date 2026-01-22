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
def get_health_report(self, category, uri, subsystems):
    result = {}
    health = {}
    status = 'Status'
    response = self.get_request(self.root_uri + uri)
    if response['ret'] is False:
        return response
    result['ret'] = True
    data = response['data']
    health[category] = {status: data.get(status, 'Status not available')}
    for sub in subsystems:
        d = None
        if sub.startswith('Links.'):
            sub = sub[len('Links.'):]
            d = data.get('Links', {})
        elif '.' in sub:
            p, sub = sub.split('.')
            u = data.get(p, {}).get('@odata.id')
            if u:
                r = self.get_request(self.root_uri + u)
                if r['ret']:
                    d = r['data']
            if not d:
                continue
        else:
            d = data
        health[sub] = []
        self.get_health_subsystem(sub, d, health)
        if not health[sub]:
            del health[sub]
    result['entries'] = health
    return result