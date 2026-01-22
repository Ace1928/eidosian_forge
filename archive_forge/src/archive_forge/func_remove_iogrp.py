from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible.module_utils._text import to_native
def remove_iogrp(self, list_of_iogrp):
    self.restapi.svc_run_command('rmvdiskaccess', {'iogrp': ':'.join(list_of_iogrp)}, [self.name])
    self.changed = True