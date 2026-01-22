from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
import re
import traceback
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
def module_replaces(self, new_module, module):
    self.module_deprecated(module)
    module.warn('netapp.ontap.%s should be used instead.' % new_module)