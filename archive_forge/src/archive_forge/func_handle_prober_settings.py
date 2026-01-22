from __future__ import absolute_import, division, print_function
import re
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def handle_prober_settings(self):
    if self.want.prober_preference == 'pool' and self.want.prober_pool is None:
        raise F5ModuleError("A prober_pool needs to be set if prober_preference is set to 'pool'")
    if self.want.prober_preference is not None and self.want.prober_fallback is not None:
        if self.want.prober_preference == self.want.prober_fallback:
            raise F5ModuleError('The parameters for prober_preference and prober_fallback must not be the same.')
    if self.want.prober_fallback == 'pool' and self.want.prober_pool is None:
        raise F5ModuleError("A prober_pool needs to be set if prober_fallback is set to 'pool'")