from __future__ import absolute_import, division, print_function
import shlex
import time
import traceback
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils.common import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible.module_utils.basic import human_to_bytes
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_text, to_native
def has_healthcheck_changed(self, old_publish):
    if self.healthcheck_disabled is False and self.healthcheck is None:
        return False
    if self.healthcheck_disabled:
        if old_publish.healthcheck is None:
            return False
        if old_publish.healthcheck.get('test') == ['NONE']:
            return False
    return self.healthcheck != old_publish.healthcheck