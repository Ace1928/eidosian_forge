from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible.module_utils._text import to_native
def parameter_handling_while_renaming(self):
    parameters = {'ownershipgroup': self.ownershipgroup, 'noownershipgroup': self.noownershipgroup, 'porttype': self.porttype, 'portset_type': self.portset_type}
    parameters_exists = [parameter for parameter, value in parameters.items() if value]
    if parameters_exists:
        self.module.fail_json(msg='Parameters {0} not supported while renaming a portset.'.format(', '.join(parameters_exists)))