from __future__ import (absolute_import, division, print_function)
import re
from abc import ABC, abstractmethod
from ansible.errors import AnsibleConnectionFailure
def on_deauthorize(self):
    """Deprecated method for privilege deescalation
        """
    return self.on_unbecome()