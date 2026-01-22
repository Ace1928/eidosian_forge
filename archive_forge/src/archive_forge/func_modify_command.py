from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import svc_ssh_argument_spec, get_logger
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_ssh import IBMSVCssh
from ansible.module_utils._text import to_native
def modify_command(self, argument):
    index = None
    command = [item.strip() for item in argument.split()]
    if command:
        for n, word in enumerate(command):
            if word.startswith('ls') and 'svcinfo' in command[n - 1]:
                index = n
                break
    if index:
        command.insert(index + 1, '-json')
    return ' '.join(command)