from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.six import binary_type, text_type
from ansible.module_utils.common._collections_compat import MutableMapping, MutableSequence
from ansible.plugins.callback.default import CallbackModule as CallbackModule_default
from ansible.utils.color import colorize, hostcolor
from ansible.utils.display import Display
import sys
def v2_playbook_item_on_skipped(self, result):
    self.v2_runner_item_on_skipped(result)