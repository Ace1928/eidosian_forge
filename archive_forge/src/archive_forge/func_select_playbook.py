from __future__ import (absolute_import, division, print_function)
from ansible.cli import CLI
import datetime
import os
import platform
import random
import shlex
import shutil
import socket
import sys
import time
from ansible import constants as C
from ansible import context
from ansible.cli.arguments import option_helpers as opt_help
from ansible.errors import AnsibleOptionsError
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.plugins.loader import module_loader
from ansible.utils.cmd_functions import run_cmd
from ansible.utils.display import Display
@staticmethod
def select_playbook(path):
    playbook = None
    errors = []
    if context.CLIARGS['args'] and context.CLIARGS['args'][0] is not None:
        playbooks = []
        for book in context.CLIARGS['args']:
            book_path = os.path.join(path, book)
            rc = PullCLI.try_playbook(book_path)
            if rc != 0:
                errors.append('%s: %s' % (book_path, PullCLI.PLAYBOOK_ERRORS[rc]))
                continue
            playbooks.append(book_path)
        if 0 < len(errors):
            display.warning('\n'.join(errors))
        elif len(playbooks) == len(context.CLIARGS['args']):
            playbook = ' '.join(playbooks)
        return playbook
    else:
        fqdn = socket.getfqdn()
        hostpb = os.path.join(path, fqdn + '.yml')
        shorthostpb = os.path.join(path, fqdn.split('.')[0] + '.yml')
        localpb = os.path.join(path, PullCLI.DEFAULT_PLAYBOOK)
        for pb in [hostpb, shorthostpb, localpb]:
            rc = PullCLI.try_playbook(pb)
            if rc == 0:
                playbook = pb
                break
            else:
                errors.append('%s: %s' % (pb, PullCLI.PLAYBOOK_ERRORS[rc]))
        if playbook is None:
            display.warning('\n'.join(errors))
        return playbook