from __future__ import absolute_import, division, print_function
from . import utils
def validate_binding_module_params(params):
    if params['state'] == 'present':
        if not (params['users'] or params['groups']):
            return 'missing required parameters: users or groups'