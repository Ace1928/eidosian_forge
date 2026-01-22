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
def get_health_subsystem(self, subsystem, data, health):
    if subsystem in data:
        sub = data.get(subsystem)
        if isinstance(sub, list):
            for r in sub:
                if '@odata.id' in r:
                    uri = r.get('@odata.id')
                    expanded = None
                    if '#' in uri and len(r) > 1:
                        expanded = r
                    self.get_health_resource(subsystem, uri, health, expanded)
        elif isinstance(sub, dict):
            if '@odata.id' in sub:
                uri = sub.get('@odata.id')
                self.get_health_resource(subsystem, uri, health, None)
    elif 'Members' in data:
        for m in data.get('Members'):
            u = m.get('@odata.id')
            r = self.get_request(self.root_uri + u)
            if r.get('ret'):
                d = r.get('data')
                self.get_health_subsystem(subsystem, d, health)