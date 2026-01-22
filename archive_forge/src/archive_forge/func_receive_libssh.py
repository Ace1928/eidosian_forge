from __future__ import absolute_import, division, print_function
import getpass
import json
import logging
import os
import re
import signal
import socket
import time
import traceback
from functools import wraps
from io import BytesIO
from ansible.errors import AnsibleConnectionFailure, AnsibleError
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import PY3
from ansible.module_utils.six.moves import cPickle
from ansible.playbook.play_context import PlayContext
from ansible.plugins.loader import cache_loader, cliconf_loader, connection_loader, terminal_loader
from ansible_collections.ansible.netcommon.plugins.connection.libssh import HAS_PYLIBSSH
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
from ansible_collections.ansible.netcommon.plugins.plugin_utils.connection_base import (
def receive_libssh(self, command=None, prompts=None, answer=None, newline=True, prompt_retry_check=False, check_all=False, strip_prompt=True):
    self._command_response = resp = b''
    command_prompt_matched = False
    handled = False
    errored_response = None
    while True:
        if command_prompt_matched:
            data = self._read_post_command_prompt_match()
            if data:
                command_prompt_matched = False
            else:
                return self._command_response
        else:
            try:
                data = self._ssh_shell.read_bulk_response()
            except OSError:
                break
        if not data:
            continue
        self._last_recv_window = self._strip(data)
        resp += self._last_recv_window
        self._window_count += 1
        self._log_messages('response-%s: %s' % (self._window_count, data))
        if prompts and (not handled):
            handled = self._handle_prompt(resp, prompts, answer, newline, False, check_all)
            self._matched_prompt_window = self._window_count
        elif prompts and handled and prompt_retry_check and (self._matched_prompt_window + 1 == self._window_count):
            if self._handle_prompt(resp, prompts, answer, newline, prompt_retry_check, check_all):
                raise AnsibleConnectionFailure("For matched prompt '%s', answer is not valid" % self._matched_cmd_prompt)
        if self._find_error(resp):
            errored_response = resp
        if self._find_prompt(resp):
            if errored_response:
                raise AnsibleConnectionFailure(errored_response)
            self._last_response = data
            self._command_response = self._sanitize(resp, command, strip_prompt)
            command_prompt_matched = True