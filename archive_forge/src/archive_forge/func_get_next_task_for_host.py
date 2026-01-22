from __future__ import (absolute_import, division, print_function)
import fnmatch
from enum import IntEnum, IntFlag
from ansible import constants as C
from ansible.errors import AnsibleAssertionError
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.playbook.block import Block
from ansible.playbook.task import Task
from ansible.utils.display import Display
def get_next_task_for_host(self, host, peek=False):
    display.debug('getting the next task for host %s' % host.name)
    s = self.get_host_state(host)
    task = None
    if s.run_state == IteratingStates.COMPLETE:
        display.debug('host %s is done iterating, returning' % host.name)
        return (s, None)
    s, task = self._get_next_task_from_state(s, host=host)
    if not peek:
        self.set_state_for_host(host.name, s)
    display.debug('done getting next task for host %s' % host.name)
    display.debug(' ^ task is: %s' % task)
    display.debug(' ^ state is: %s' % s)
    return (s, task)