from __future__ import absolute_import, division, print_function
import hashlib
import os
import posixpath
import shutil
import io
import tempfile
import traceback
import re
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.ansible_release import __version__ as ansible_version
from re import match
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
def is_invalid_checksum(self, file, remote_url, checksum_alg='md5'):
    if os.path.exists(file):
        local_checksum = self._local_checksum(checksum_alg, file)
        if self.local:
            parsed_url = urlparse(remote_url)
            remote_checksum = self._local_checksum(checksum_alg, parsed_url.path)
        else:
            try:
                remote_checksum = to_text(self._getContent(remote_url + '.' + checksum_alg, 'Failed to retrieve checksum', False), errors='strict')
            except UnicodeError as e:
                return 'Cannot retrieve a valid %s checksum from %s: %s' % (checksum_alg, remote_url, to_native(e))
            if not remote_checksum:
                return 'Cannot find %s checksum from %s' % (checksum_alg, remote_url)
        try:
            _remote_checksum = remote_checksum.split(None, 1)[0]
            remote_checksum = _remote_checksum
        except IndexError:
            pass
        if local_checksum.lower() == remote_checksum.lower():
            return None
        else:
            return 'Checksum does not match: we computed ' + local_checksum + ' but the repository states ' + remote_checksum
    return 'Path does not exist: ' + file