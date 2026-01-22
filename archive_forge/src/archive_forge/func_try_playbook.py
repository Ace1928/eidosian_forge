from __future__ import (absolute_import, division, print_function)
from ansible.cli import CLI
import datetime
import os
import platform
import random
import shlex
import shutil
import socket
import sys
import time
from ansible import constants as C
from ansible import context
from ansible.cli.arguments import option_helpers as opt_help
from ansible.errors import AnsibleOptionsError
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.plugins.loader import module_loader
from ansible.utils.cmd_functions import run_cmd
from ansible.utils.display import Display
@staticmethod
def try_playbook(path):
    if not os.path.exists(path):
        return 1
    if not os.access(path, os.R_OK):
        return 2
    return 0