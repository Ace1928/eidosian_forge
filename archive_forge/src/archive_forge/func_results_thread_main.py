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
def results_thread_main(strategy):
    while True:
        try:
            result = strategy._final_q.get()
            if isinstance(result, StrategySentinel):
                break
            elif isinstance(result, DisplaySend):
                dmethod = getattr(display, result.method)
                dmethod(*result.args, **result.kwargs)
            elif isinstance(result, CallbackSend):
                for arg in result.args:
                    if isinstance(arg, TaskResult):
                        strategy.normalize_task_result(arg)
                        break
                strategy._tqm.send_callback(result.method_name, *result.args, **result.kwargs)
            elif isinstance(result, TaskResult):
                strategy.normalize_task_result(result)
                with strategy._results_lock:
                    strategy._results.append(result)
            elif isinstance(result, PromptSend):
                try:
                    value = display.prompt_until(result.prompt, private=result.private, seconds=result.seconds, complete_input=result.complete_input, interrupt_input=result.interrupt_input)
                except AnsibleError as e:
                    value = e
                except BaseException as e:
                    try:
                        raise AnsibleError(f'{e}') from e
                    except AnsibleError as e:
                        value = e
                strategy._workers[result.worker_id].worker_queue.put(value)
            else:
                display.warning('Received an invalid object (%s) in the result queue: %r' % (type(result), result))
        except (IOError, EOFError):
            break
        except queue.Empty:
            pass