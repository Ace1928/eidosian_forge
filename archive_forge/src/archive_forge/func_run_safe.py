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
def run_safe(self):
    while True:
        try:
            return self.run()
        except APIError as e:
            if self.retries > 0 and 'update out of sequence' in str(e.explanation):
                self.retries -= 1
                time.sleep(1)
            else:
                raise