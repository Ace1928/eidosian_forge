from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
import re
import traceback
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
def module_deprecated(self, module):
    module.warn(ZAPI_ONLY_DEPRECATION_MESSAGE)