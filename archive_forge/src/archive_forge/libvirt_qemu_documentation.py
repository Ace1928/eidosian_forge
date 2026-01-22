from __future__ import (absolute_import, division, print_function)
import base64
import json
import shlex
import time
import traceback
from ansible.errors import AnsibleError, AnsibleConnectionFailure, AnsibleFileNotFound
from ansible.module_utils._text import to_bytes, to_native, to_text
from ansible.module_utils.six import raise_from
from ansible.plugins.connection import ConnectionBase, BUFSIZE
from ansible.plugins.shell.powershell import _parse_clixml
from ansible.utils.display import Display
from functools import partial
from os.path import exists
 terminate the connection; nothing to do here 