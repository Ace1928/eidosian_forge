from __future__ import absolute_import, division, print_function
import copy
import os
import ssl
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.connection import exec_command
from ..module_utils.common import (
def update_certificate(self):
    self.create_csr()
    cmd = 'openssl x509 -req -in {0}/ssl.csr/{3}.csr -signkey {0}/ssl.key/{2} -days {4} -out {0}/ssl.crt/{1}'.format('/config/httpd/conf', self.want.cert_name, self.want.key_name, os.path.splitext(self.want.cert_name)[0], self.want.days_valid)
    rc, out, err = exec_command(self.module, cmd)
    if rc != 0:
        raise F5ModuleError(err)