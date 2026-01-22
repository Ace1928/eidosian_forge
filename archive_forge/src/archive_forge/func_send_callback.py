from __future__ import (absolute_import, division, print_function)
import os
import sys
import tempfile
import threading
import time
import typing as t
import multiprocessing.queues
from ansible import constants as C
from ansible import context
from ansible.errors import AnsibleError
from ansible.executor.play_iterator import PlayIterator
from ansible.executor.stats import AggregateStats
from ansible.executor.task_result import TaskResult
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_text, to_native
from ansible.playbook.play_context import PlayContext
from ansible.playbook.task import Task
from ansible.plugins.loader import callback_loader, strategy_loader, module_loader
from ansible.plugins.callback import CallbackBase
from ansible.template import Templar
from ansible.vars.hostvars import HostVars
from ansible.vars.reserved import warn_if_reserved
from ansible.utils.display import Display
from ansible.utils.lock import lock_decorator
from ansible.utils.multiprocessing import context as multiprocessing_context
from dataclasses import dataclass
@lock_decorator(attr='_callback_lock')
def send_callback(self, method_name, *args, **kwargs):
    for callback_plugin in [self._stdout_callback] + self._callback_plugins:
        if getattr(callback_plugin, 'disabled', False):
            continue
        wants_implicit_tasks = getattr(callback_plugin, 'wants_implicit_tasks', False)
        methods = []
        for possible in [method_name, 'v2_on_any']:
            gotit = getattr(callback_plugin, possible, None)
            if gotit is None:
                gotit = getattr(callback_plugin, possible.removeprefix('v2_'), None)
            if gotit is not None:
                methods.append(gotit)
        new_args = []
        is_implicit_task = False
        for arg in args:
            if isinstance(arg, TaskResult):
                new_args.append(arg.clean_copy())
            else:
                new_args.append(arg)
            if isinstance(arg, Task) and arg.implicit:
                is_implicit_task = True
        if is_implicit_task and (not wants_implicit_tasks):
            continue
        for method in methods:
            try:
                method(*new_args, **kwargs)
            except Exception as e:
                display.warning(u'Failure using method (%s) in callback plugin (%s): %s' % (to_text(method_name), to_text(callback_plugin), to_text(e)))
                from traceback import format_tb
                from sys import exc_info
                display.vvv('Callback Exception: \n' + ' '.join(format_tb(exc_info()[2])))