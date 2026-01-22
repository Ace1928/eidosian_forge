from __future__ import absolute_import, division, print_function
from os.path import isfile
from os import getuid, unlink
import re
import shutil
import tempfile
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six.moves import configparser
from ansible.module_utils import distro
def sync_syspurpose(self):
    """
        Try to synchronize syspurpose attributes with server
        """
    args = [SUBMAN_CMD, 'status']
    rc, stdout, stderr = self.module.run_command(args, check_rc=False)