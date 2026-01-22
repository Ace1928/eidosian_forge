from __future__ import absolute_import, division, print_function
import os
import tempfile
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import string_types
from ansible.module_utils._text import to_bytes
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def validate_shell_parameter(self):
    """Method to validate shell parameters.

        Raise when shell attribute is set to 'bash' with roles set to
        either 'admin' or 'resource-admin'.

        NOTE: Admin and Resource-Admin roles automatically enable access to
        all partitions, removing any other roles that the user might have
        had. There are few other roles which do that but those roles,
        do not allow bash.
        """
    err = "Shell access is only available to 'admin' or 'resource-admin' roles."
    permit = ['admin', 'resource-admin']
    if self.want.partition_access is not None:
        want = self.want.partition_access
        if not any((r['role'] for r in want if r['role'] in permit)):
            raise F5ModuleError(err)