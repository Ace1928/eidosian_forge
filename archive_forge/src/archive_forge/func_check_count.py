from __future__ import absolute_import, division, print_function
import os
import os.path
import re
import shutil
import subprocess
import tempfile
import time
import shlex
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.parsing.convert_bool import BOOLEANS_FALSE
from ansible.module_utils.common.text.converters import to_text, to_bytes
def check_count(self, count, method):
    if count > 1:
        self.failure(error='Failed to %s container' % method, rc=1, msg='The container [ %s ] failed to %s. Check to lxc is available and that the container is in a functional state.' % (self.container_name, method))