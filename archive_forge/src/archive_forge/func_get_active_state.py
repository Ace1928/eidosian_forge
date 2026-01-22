from __future__ import (absolute_import, division, print_function)
import fnmatch
from enum import IntEnum, IntFlag
from ansible import constants as C
from ansible.errors import AnsibleAssertionError
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.playbook.block import Block
from ansible.playbook.task import Task
from ansible.utils.display import Display
def get_active_state(self, state):
    """
        Finds the active state, recursively if necessary when there are child states.
        """
    if state.run_state == IteratingStates.TASKS and state.tasks_child_state is not None:
        return self.get_active_state(state.tasks_child_state)
    elif state.run_state == IteratingStates.RESCUE and state.rescue_child_state is not None:
        return self.get_active_state(state.rescue_child_state)
    elif state.run_state == IteratingStates.ALWAYS and state.always_child_state is not None:
        return self.get_active_state(state.always_child_state)
    return state