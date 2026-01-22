from __future__ import (absolute_import, division, print_function)
import re
from abc import ABC, abstractmethod
from ansible.errors import AnsibleConnectionFailure
def on_authorize(self, passwd=None):
    """Deprecated method for privilege escalation

        :kwarg passwd: String containing the password
        """
    return self.on_become(passwd)