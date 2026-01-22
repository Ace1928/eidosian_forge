from __future__ import absolute_import, division, print_function
import base64
import binascii
import json
import mimetypes
import os
import random
import string
import traceback
from ansible_collections.community.general.plugins.module_utils.module_helper import StateModuleHelper, cause_changes
from ansible.module_utils.six.moves.urllib.request import pathname2url
from ansible.module_utils.common.text.converters import to_text, to_bytes, to_native
from ansible.module_utils.urls import fetch_url
@cause_changes(on_success=True)
def operation_attach(self):
    v = self.vars
    filename = v.attachment.get('filename')
    content = v.attachment.get('content')
    if not any((filename, content)):
        raise ValueError('at least one of filename or content must be provided')
    mime = v.attachment.get('mimetype')
    if not os.path.isfile(filename):
        raise ValueError('The provided filename does not exist: %s' % filename)
    content_type, data = self._prepare_attachment(filename, content, mime)
    url = v.restbase + '/issue/' + v.issue + '/attachments'
    return (True, self.post(url, data, content_type=content_type, additional_headers={'X-Atlassian-Token': 'no-check'}))