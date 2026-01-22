from __future__ import absolute_import, division, print_function
import copy
import os
import ssl
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.connection import exec_command
from ..module_utils.common import (
def generate_new(self):
    self.generate_cert_key()
    if self.want.cert_name != 'server.crt' or self.want.key_name != 'server.key':
        self.configure_new_cert()
    self.restart_daemon()
    if self.want.add_to_trusted:
        self.copy_files_to_trusted()
    return True