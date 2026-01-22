from __future__ import (absolute_import, division, print_function)
import cmd
import functools
import os
import pprint
import queue
import sys
import threading
import time
import typing as t
from collections import deque
from multiprocessing import Lock
from jinja2.exceptions import UndefinedError
from ansible import constants as C
from ansible import context
from ansible.errors import AnsibleError, AnsibleFileNotFound, AnsibleUndefinedVariable, AnsibleParserError
from ansible.executor import action_write_locks
from ansible.executor.play_iterator import IteratingStates, PlayIterator
from ansible.executor.process.worker import WorkerProcess
from ansible.executor.task_result import TaskResult
from ansible.executor.task_queue_manager import CallbackSend, DisplaySend, PromptSend
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.connection import Connection, ConnectionError
from ansible.playbook.conditional import Conditional
from ansible.playbook.handler import Handler
from ansible.playbook.helpers import load_list_of_blocks
from ansible.playbook.task import Task
from ansible.playbook.task_include import TaskInclude
from ansible.plugins import loader as plugin_loader
from ansible.template import Templar
from ansible.utils.display import Display
from ansible.utils.fqcn import add_internal_fqcns
from ansible.utils.unsafe_proxy import wrap_var
from ansible.utils.sentinel import Sentinel
from ansible.utils.vars import combine_vars, isidentifier
from ansible.vars.clean import strip_internal_keys, module_response_deepcopy
def debug_closure(func):
    """Closure to wrap ``StrategyBase._process_pending_results`` and invoke the task debugger"""

    @functools.wraps(func)
    def inner(self, iterator, one_pass=False, max_passes=None):
        status_to_stats_map = (('is_failed', 'failures'), ('is_unreachable', 'dark'), ('is_changed', 'changed'), ('is_skipped', 'skipped'))
        prev_host_states = iterator.host_states.copy()
        results = func(self, iterator, one_pass=one_pass, max_passes=max_passes)
        _processed_results = []
        for result in results:
            task = result._task
            host = result._host
            _queued_task_args = self._queued_task_cache.pop((host.name, task._uuid), None)
            task_vars = _queued_task_args['task_vars']
            play_context = _queued_task_args['play_context']
            try:
                prev_host_state = prev_host_states[host.name]
            except KeyError:
                prev_host_state = iterator.get_host_state(host)
            while result.needs_debugger(globally_enabled=self.debugger_active):
                next_action = NextAction()
                dbg = Debugger(task, host, task_vars, play_context, result, next_action)
                dbg.cmdloop()
                if next_action.result == NextAction.REDO:
                    self._tqm.clear_failed_hosts()
                    if task.run_once and iterator._play.strategy in add_internal_fqcns(('linear',)) and result.is_failed():
                        for host_name, state in prev_host_states.items():
                            if host_name == host.name:
                                continue
                            iterator.set_state_for_host(host_name, state)
                            iterator._play._removed_hosts.remove(host_name)
                    iterator.set_state_for_host(host.name, prev_host_state)
                    for method, what in status_to_stats_map:
                        if getattr(result, method)():
                            self._tqm._stats.decrement(what, host.name)
                    self._tqm._stats.decrement('ok', host.name)
                    self._queue_task(host, task, task_vars, play_context)
                    _processed_results.extend(debug_closure(func)(self, iterator, one_pass))
                    break
                elif next_action.result == NextAction.CONTINUE:
                    _processed_results.append(result)
                    break
                elif next_action.result == NextAction.EXIT:
                    sys.exit(99)
            else:
                _processed_results.append(result)
        return _processed_results
    return inner