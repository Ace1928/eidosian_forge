from __future__ import (absolute_import, division, print_function)
import os
from subprocess import call, Popen, PIPE
from ansible.errors import AnsibleError, AnsibleConnectionFailure, AnsibleFileNotFound
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils._text import to_bytes, to_text
from ansible.plugins.connection import ConnectionBase
 close the connection (nothing to do here) 