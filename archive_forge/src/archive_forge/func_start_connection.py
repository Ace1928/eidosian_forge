from __future__ import (absolute_import, division, print_function)
import os
import pty
import time
import json
import signal
import subprocess
import sys
import termios
import traceback
from ansible import constants as C
from ansible.errors import AnsibleError, AnsibleParserError, AnsibleUndefinedVariable, AnsibleConnectionFailure, AnsibleActionFail, AnsibleActionSkip
from ansible.executor.task_result import TaskResult
from ansible.executor.module_common import get_action_args_with_defaults
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import binary_type
from ansible.module_utils.common.text.converters import to_text, to_native
from ansible.module_utils.connection import write_to_file_descriptor
from ansible.playbook.conditional import Conditional
from ansible.playbook.task import Task
from ansible.plugins import get_plugin_class
from ansible.plugins.loader import become_loader, cliconf_loader, connection_loader, httpapi_loader, netconf_loader, terminal_loader
from ansible.template import Templar
from ansible.utils.collection_loader import AnsibleCollectionConfig
from ansible.utils.listify import listify_lookup_plugin_terms
from ansible.utils.unsafe_proxy import to_unsafe_text, wrap_var
from ansible.vars.clean import namespace_facts, clean_facts
from ansible.utils.display import Display
from ansible.utils.vars import combine_vars, isidentifier
def start_connection(play_context, options, task_uuid):
    """
    Starts the persistent connection
    """
    candidate_paths = [C.ANSIBLE_CONNECTION_PATH or os.path.dirname(sys.argv[0])]
    candidate_paths.extend(os.environ.get('PATH', '').split(os.pathsep))
    for dirname in candidate_paths:
        ansible_connection = os.path.join(dirname, 'ansible-connection')
        if os.path.isfile(ansible_connection):
            display.vvvv('Found ansible-connection at path {0}'.format(ansible_connection))
            break
    else:
        raise AnsibleError("Unable to find location of 'ansible-connection'. Please set or check the value of ANSIBLE_CONNECTION_PATH")
    env = os.environ.copy()
    env.update({'ANSIBLE_BECOME_PLUGINS': become_loader.print_paths(), 'ANSIBLE_CLICONF_PLUGINS': cliconf_loader.print_paths(), 'ANSIBLE_COLLECTIONS_PATH': to_native(os.pathsep.join(AnsibleCollectionConfig.collection_paths)), 'ANSIBLE_CONNECTION_PLUGINS': connection_loader.print_paths(), 'ANSIBLE_HTTPAPI_PLUGINS': httpapi_loader.print_paths(), 'ANSIBLE_NETCONF_PLUGINS': netconf_loader.print_paths(), 'ANSIBLE_TERMINAL_PLUGINS': terminal_loader.print_paths()})
    verbosity = []
    if display.verbosity:
        verbosity.append('-%s' % ('v' * display.verbosity))
    python = sys.executable
    master, slave = pty.openpty()
    p = subprocess.Popen([python, ansible_connection, *verbosity, to_text(os.getppid()), to_text(task_uuid)], stdin=slave, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    os.close(slave)
    old = termios.tcgetattr(master)
    new = termios.tcgetattr(master)
    new[3] = new[3] & ~termios.ICANON
    try:
        termios.tcsetattr(master, termios.TCSANOW, new)
        write_to_file_descriptor(master, options)
        write_to_file_descriptor(master, play_context.serialize())
        stdout, stderr = p.communicate()
    finally:
        termios.tcsetattr(master, termios.TCSANOW, old)
    os.close(master)
    if p.returncode == 0:
        result = json.loads(to_text(stdout, errors='surrogate_then_replace'))
    else:
        try:
            result = json.loads(to_text(stderr, errors='surrogate_then_replace'))
        except json.decoder.JSONDecodeError:
            result = {'error': to_text(stderr, errors='surrogate_then_replace')}
    if 'messages' in result:
        for level, message in result['messages']:
            if level == 'log':
                display.display(message, log_only=True)
            elif level in ('debug', 'v', 'vv', 'vvv', 'vvvv', 'vvvvv', 'vvvvvv'):
                getattr(display, level)(message, host=play_context.remote_addr)
            elif hasattr(display, level):
                getattr(display, level)(message)
            else:
                display.vvvv(message, host=play_context.remote_addr)
    if 'error' in result:
        if display.verbosity > 2:
            if result.get('exception'):
                msg = 'The full traceback is:\n' + result['exception']
                display.display(msg, color=C.COLOR_ERROR)
        raise AnsibleError(result['error'])
    return result['socket_path']