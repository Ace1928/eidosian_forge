from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
import re
import traceback
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
def ignore_missing_vserver_on_delete(self, error, vserver_name=None):
    """ When a resource is expected to be absent, it's OK if the containing vserver is also absent.
            This function expects self.parameters('vserver') to be set or the vserver_name argument to be passed.
            error is an error returned by rest_generic.get_xxxx.
        """
    if self.parameters.get('state') != 'absent':
        return False
    if vserver_name is None:
        if self.parameters.get('vserver') is None:
            self.ansible_module.fail_json(msg='Internal error, vserver name is required, when processing error: %s' % error, exception=traceback.format_exc())
        vserver_name = self.parameters['vserver']
    if isinstance(error, str):
        pass
    elif isinstance(error, dict):
        if 'message' in error:
            error = error['message']
        else:
            self.ansible_module.fail_json(msg='Internal error, error should contain "message" key, found: %s' % error, exception=traceback.format_exc())
    else:
        self.ansible_module.fail_json(msg='Internal error, error should be str or dict, found: %s, %s' % (type(error), error), exception=traceback.format_exc())
    return 'SVM "%s" does not exist.' % vserver_name in error