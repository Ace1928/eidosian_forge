from __future__ import (absolute_import, division, print_function)
import abc
import os
import json
import subprocess
from ansible.plugins.lookup import LookupBase
from ansible.errors import AnsibleLookupError, AnsibleOptionsError
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.module_utils.six import with_metaclass
from ansible_collections.community.general.plugins.module_utils.onepassword import OnePasswordConfig
def set_token(self):
    if self._config.config_file_path and os.path.isfile(self._config.config_file_path):
        try:
            rc, out, err = self._cli.signin()
        except AnsibleLookupError as exc:
            test_strings = ('missing required parameters', 'unauthorized')
            if any((string in exc.message.lower() for string in test_strings)):
                raise
            rc, out, err = self._cli.full_signin()
        self.token = out.strip()
    else:
        rc, out, err = self._cli.full_signin()
        self.token = out.strip()