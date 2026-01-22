from __future__ import absolute_import, division, print_function
import getpass
import json
import logging
import os
import re
import signal
import socket
import time
import traceback
from functools import wraps
from io import BytesIO
from ansible.errors import AnsibleConnectionFailure, AnsibleError
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import PY3
from ansible.module_utils.six.moves import cPickle
from ansible.playbook.play_context import PlayContext
from ansible.plugins.loader import cache_loader, cliconf_loader, connection_loader, terminal_loader
from ansible_collections.ansible.netcommon.plugins.connection.libssh import HAS_PYLIBSSH
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
from ansible_collections.ansible.netcommon.plugins.plugin_utils.connection_base import (
@property
def ssh_type_conn(self):
    if self._ssh_type_conn is None:
        if self.ssh_type == 'libssh':
            connection_plugin = 'ansible.netcommon.libssh'
        elif self.ssh_type == 'paramiko':
            connection_plugin = 'paramiko'
        else:
            raise AnsibleConnectionFailure("Invalid value '%s' set for ssh_type option. Expected value is either 'libssh' or 'paramiko'" % self._ssh_type)
        self._ssh_type_conn = connection_loader.get(connection_plugin, self._play_context, '/dev/null')
    return self._ssh_type_conn