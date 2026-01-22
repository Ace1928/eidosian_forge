from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
def host_rename(self, host_data):
    msg = ''
    self.parameter_handling_while_renaming()
    old_host_data = self.get_existing_host(self.old_name)
    if not old_host_data and (not host_data):
        self.module.fail_json(msg='Host [{0}] does not exists.'.format(self.old_name))
    elif old_host_data and host_data:
        self.module.fail_json(msg='Host [{0}] already exists.'.format(self.name))
    elif not old_host_data and host_data:
        msg = 'Host with name [{0}] already exists.'.format(self.name)
    elif old_host_data and (not host_data):
        if self.module.check_mode:
            self.changed = True
            return
        self.restapi.svc_run_command('chhost', {'name': self.name}, [self.old_name])
        self.changed = True
        msg = 'Host [{0}] has been successfully rename to [{1}].'.format(self.old_name, self.name)
    return msg