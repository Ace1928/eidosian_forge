from __future__ import (absolute_import, division, print_function)
from abc import abstractmethod
from functools import wraps
from ansible.plugins import AnsiblePlugin
from ansible.errors import AnsibleError, AnsibleConnectionFailure
from ansible.module_utils.common.text.converters import to_bytes, to_text
def reset_history(self):
    """ Resets the history of run commands
        :return: None
        """
    self.history = list()