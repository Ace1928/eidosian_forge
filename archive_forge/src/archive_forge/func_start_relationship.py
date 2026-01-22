from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
def start_relationship(self):
    """Start the migration relationship copy process."""
    cmdopts = {}
    if self.module.check_mode:
        self.changed = True
        return
    result = self.restapi.svc_run_command(cmd='startrcrelationship', cmdopts=cmdopts, cmdargs=[self.relationship_name])
    if result == '':
        self.changed = True
        self.log('succeeded to start the remote copy %s', self.relationship_name)
    elif 'message' in result:
        self.changed = True
        self.log('start the rcrelationship %s with result message %s', self.relationship_name, result['message'])
    else:
        msg = 'Failed to start the rcrelationship [%s]' % self.relationship_name
        self.module.fail_json(msg=msg)