from __future__ import (absolute_import, division, print_function)
from ansible.cli import CLI
import atexit
import cmd
import getpass
import readline
import os
import sys
from ansible import constants as C
from ansible import context
from ansible.cli.arguments import option_helpers as opt_help
from ansible.executor.task_queue_manager import TaskQueueManager
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.parsing.splitter import parse_kv
from ansible.playbook.play import Play
from ansible.plugins.list import list_plugins
from ansible.plugins.loader import module_loader, fragment_loader
from ansible.utils import plugin_docs
from ansible.utils.color import stringc
from ansible.utils.display import Display
def do_timeout(self, arg):
    """Set the timeout"""
    if arg:
        try:
            timeout = int(arg)
            if timeout < 0:
                display.error('The timeout must be greater than or equal to 1, use 0 to disable')
            else:
                self.task_timeout = timeout
        except (TypeError, ValueError) as e:
            display.error('The timeout must be a valid positive integer, or 0 to disable: %s' % to_text(e))
    else:
        self.usage_timeout()