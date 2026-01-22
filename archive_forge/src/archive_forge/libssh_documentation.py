from __future__ import absolute_import, division, print_function
import logging
import os
import re
import socket
import sys
from termios import TCIFLUSH, tcflush
from ansible.errors import AnsibleConnectionFailure, AnsibleError, AnsibleFileNotFound
from ansible.module_utils._text import to_bytes, to_native, to_text
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six.moves import input
from ansible.plugins.connection import ConnectionBase
from ansible.utils.display import Display
from ansible_collections.ansible.netcommon.plugins.plugin_utils.version import Version
terminate the connection