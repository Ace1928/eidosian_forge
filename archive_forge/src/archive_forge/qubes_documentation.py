from __future__ import (absolute_import, division, print_function)
import subprocess
from ansible.module_utils.common.text.converters import to_bytes
from ansible.plugins.connection import ConnectionBase, ensure_connect
from ansible.errors import AnsibleConnectionFailure
from ansible.utils.display import Display
 Closing the connection 