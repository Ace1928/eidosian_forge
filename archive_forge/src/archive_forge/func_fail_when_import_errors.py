from __future__ import (absolute_import, division, print_function)
import sys
def fail_when_import_errors(self, import_errors, has_azure_mgmt_netapp=True):
    if has_azure_mgmt_netapp and (not import_errors):
        return
    msg = ''
    if not has_azure_mgmt_netapp:
        msg = 'The python azure-mgmt-netapp package is required.  '
    if hasattr(self, 'module'):
        msg += 'Import errors: %s' % str(import_errors)
        self.module.fail_json(msg=msg)
    msg += str(import_errors)
    raise ImportError(msg)