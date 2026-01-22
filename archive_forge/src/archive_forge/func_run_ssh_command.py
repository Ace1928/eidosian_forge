from __future__ import absolute_import, division, print_function
import traceback
import warnings
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
def run_ssh_command(self, command):
    """ calls SSH """
    try:
        stdin, stdout, stderr = self.client.exec_command(command)
    except paramiko.SSHException as exc:
        self.module.fail_json(msg='Error running command %s: %s' % (command, to_native(exc)), exception=traceback.format_exc())
    stdin.close()
    return (stdout, stderr)