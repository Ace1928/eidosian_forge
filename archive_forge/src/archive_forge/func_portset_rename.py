from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible.module_utils._text import to_native
def portset_rename(self, portset_data):
    msg = ''
    self.parameter_handling_while_renaming()
    old_portset_data = self.is_portset_exists(self.old_name)
    if not old_portset_data and (not portset_data):
        self.module.fail_json(msg="Portset with old name {0} doesn't exist.".format(self.old_name))
    elif old_portset_data and portset_data:
        self.module.fail_json(msg='Portset [{0}] already exists.'.format(self.name))
    elif not old_portset_data and portset_data:
        msg = 'Portset with name [{0}] already exists.'.format(self.name)
    elif old_portset_data and (not portset_data):
        if self.module.check_mode:
            self.changed = True
            return
        self.restapi.svc_run_command('chportset', {'name': self.name}, [self.old_name])
        self.changed = True
        msg = 'Portset [{0}] has been successfully rename to [{1}].'.format(self.old_name, self.name)
    return msg