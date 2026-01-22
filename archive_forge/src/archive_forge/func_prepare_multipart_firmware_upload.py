from __future__ import absolute_import, division, print_function
import json
import os
import uuid
from ansible.module_utils.urls import open_url
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.six.moves.urllib.parse import urlparse
def prepare_multipart_firmware_upload(self, filename):
    """Prepare a multipart/form-data body for OCAPI firmware upload.

        :arg filename: The name of the file to upload.
        :returns: tuple of (content_type, body) where ``content_type`` is
            the ``multipart/form-data`` ``Content-Type`` header including
            ``boundary`` and ``body`` is the prepared bytestring body

        Prepares the body to include "FirmwareFile" field with the contents of the file.
        Because some OCAPI targets do not support Base-64 encoding for multipart/form-data,
        this method sends the file as binary.
        """
    boundary = str(uuid.uuid4())
    body = '--' + boundary + '\r\n'
    body += 'Content-Disposition: form-data; name="FirmwareFile"; filename="%s"\r\n' % to_native(os.path.basename(filename))
    body += 'Content-Type: application/octet-stream\r\n\r\n'
    body_bytes = bytearray(body, 'utf-8')
    with open(filename, 'rb') as f:
        body_bytes += f.read()
    body_bytes += bytearray('\r\n--%s--' % boundary, 'utf-8')
    return ('multipart/form-data; boundary=%s' % boundary, body_bytes)