from __future__ import absolute_import, division, print_function
from copy import deepcopy
import re
import os
import ast
import datetime
import shutil
import tempfile
from ansible.module_utils.basic import json
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.six import PY3
from ansible.module_utils.six.moves import filterfalse
from ansible.module_utils.six.moves.urllib.parse import urlencode, urljoin
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_native, to_text
from ansible.module_utils.connection import Connection
from ansible_collections.cisco.mso.plugins.module_utils.constants import NDO_API_VERSION_PATH_FORMAT
def request_upload(self, path, fields=None, method='POST', api_version='v1'):
    """Generic HTTP MultiPart POST method for MSO uploads."""
    self.path = path
    if self.platform != 'nd':
        self.url = urljoin(self.baseuri, path)
    info = dict()
    if self.platform == 'nd':
        try:
            if os.path.exists(self.params.get('backup')):
                info = self.connection.send_file_request(method, NDO_API_VERSION_PATH_FORMAT.format(api_version=api_version, path=path), file=self.params.get('backup'), remote_path=self.params.get('remote_path'))
            else:
                self.fail_json(msg="Upload failed due to: No such file or directory, Backup file: '{0}'".format(self.params.get('backup')))
        except Exception as error:
            self.fail_json('NDO upload failed due to: {0}'.format(error))
    else:
        if not HAS_MULTIPART_ENCODER:
            self.fail_json(msg='requests-toolbelt is required for the upload state of this module')
        mp_encoder = MultipartEncoder(fields=fields)
        self.headers['Content-Type'] = mp_encoder.content_type
        self.headers['Accept-Encoding'] = 'gzip, deflate, br'
        resp, info = fetch_url(self.module, self.url, headers=self.headers, data=mp_encoder, method=method, timeout=self.params.get('timeout'), use_proxy=self.params.get('use_proxy'))
    self.response = info.get('msg')
    self.status = info.get('status')
    if 'modified' in info:
        self.has_modified = True
        if info.get('modified') == 'false':
            self.result['changed'] = False
        elif info.get('modified') == 'true':
            self.result['changed'] = True
    if self.status in (200, 201, 202, 204):
        if self.platform == 'nd':
            return info
        else:
            output = resp.read()
            if output:
                return json.loads(output)
    elif self.status:
        if self.status >= 400:
            try:
                if self.platform == 'nd':
                    payload = info.get('body')
                else:
                    payload = json.loads(resp.read())
            except (ValueError, AttributeError):
                try:
                    payload = json.loads(info.get('body'))
                except Exception:
                    self.fail_json(msg='MSO Error:', info=info)
            if 'code' in payload:
                self.fail_json(msg='MSO Error {code}: {message}'.format(**payload), info=info, payload=payload)
            else:
                self.fail_json(msg='MSO Error:'.format(**payload), info=info, payload=payload)
    else:
        self.fail_json(msg='Backup file upload failed due to: {0}'.format(info))
    return {}