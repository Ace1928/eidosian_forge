from __future__ import absolute_import, division, print_function
import re
import uuid
import time
from ansible.module_utils.basic import AnsibleModule

    Removes a NIC

    module : AnsibleModule object
    profitbricks: authenticated profitbricks object.

    Returns:
        True if the NIC was removed, false otherwise
    