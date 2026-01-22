from __future__ import (absolute_import, division, print_function)
import shlex
from abc import abstractmethod
from random import choice
from string import ascii_lowercase
from gettext import dgettext
from ansible.errors import AnsibleError
from ansible.module_utils.common.text.converters import to_bytes
from ansible.plugins import AnsiblePlugin
def check_missing_password(self, b_output):
    for errstring in self.missing:
        if self._check_password_error(b_output, errstring):
            return True
    return False