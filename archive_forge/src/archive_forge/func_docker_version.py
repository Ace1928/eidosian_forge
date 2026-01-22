from __future__ import (absolute_import, division, print_function)
import fcntl
import os
import os.path
import subprocess
import re
from ansible.compat import selectors
from ansible.errors import AnsibleError, AnsibleFileNotFound
from ansible.module_utils.six.moves import shlex_quote
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.plugins.connection import ConnectionBase, BUFSIZE
from ansible.utils.display import Display
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
@property
def docker_version(self):
    if not self._version:
        self._set_docker_args()
        self._version = self._get_docker_version()
        if self._version == u'dev':
            display.warning(u'Docker version number is "dev". Will assume latest version.')
        if self._version != u'dev' and LooseVersion(self._version) < LooseVersion(u'1.3'):
            raise AnsibleError('docker connection type requires docker 1.3 or higher')
    return self._version