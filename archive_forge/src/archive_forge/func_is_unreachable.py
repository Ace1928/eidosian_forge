from __future__ import (absolute_import, division, print_function)
from ansible import constants as C
from ansible.parsing.dataloader import DataLoader
from ansible.vars.clean import module_response_deepcopy, strip_internal_keys
def is_unreachable(self):
    return self._check_key('unreachable')