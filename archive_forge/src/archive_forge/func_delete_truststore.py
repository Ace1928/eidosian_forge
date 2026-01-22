from __future__ import absolute_import, division, print_function
from traceback import format_exc
import json
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_ssh import IBMSVCssh
from ansible.module_utils._text import to_native
def delete_truststore(self):
    if self.module.check_mode:
        self.changed = True
        return
    cmd = 'rmtruststore {0}'.format(self.name)
    self.log('Command to be executed: %s', cmd)
    stdin, stdout, stderr = self.ssh_client.client.exec_command(cmd)
    result = stdout.read().decode('utf-8')
    rc = stdout.channel.recv_exit_status()
    if rc > 0:
        self.log('Error in executing command: %s', cmd)
        self.raise_error(stderr)
    else:
        self.log('Truststore (%s) deleted', self.name)
        self.log(result)
        self.changed = True