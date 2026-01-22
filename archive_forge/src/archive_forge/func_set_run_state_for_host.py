from __future__ import (absolute_import, division, print_function)
import fnmatch
from enum import IntEnum, IntFlag
from ansible import constants as C
from ansible.errors import AnsibleAssertionError
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.playbook.block import Block
from ansible.playbook.task import Task
from ansible.utils.display import Display
def set_run_state_for_host(self, hostname: str, run_state: IteratingStates) -> None:
    if not isinstance(run_state, IteratingStates):
        raise AnsibleAssertionError('Expected run_state to be a IteratingStates but was %s' % type(run_state))
    self._host_states[hostname].run_state = run_state