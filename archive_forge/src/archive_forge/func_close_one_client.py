from __future__ import (absolute_import, division, print_function)
import time
import ssl
from os import environ
from ansible.module_utils.six import string_types
from ansible.module_utils.basic import AnsibleModule
def close_one_client(self):
    """
        Close the pyone session.
        """
    self.one.server_close()