from __future__ import (absolute_import, division, print_function)
import fcntl
import hashlib
import io
import os
import pickle
import signal
import socket
import sys
import time
import traceback
import errno
import json
from contextlib import contextmanager
from ansible import constants as C
from ansible.cli.arguments import option_helpers as opt_help
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.module_utils.connection import Connection, ConnectionError, send_data, recv_data
from ansible.module_utils.service import fork_process
from ansible.parsing.ajson import AnsibleJSONEncoder, AnsibleJSONDecoder
from ansible.playbook.play_context import PlayContext
from ansible.plugins.loader import connection_loader, init_plugin_loader
from ansible.utils.path import unfrackpath, makedirs_safe
from ansible.utils.display import Display
from ansible.utils.jsonrpc import JsonRpcServer
@contextmanager
def file_lock(lock_path):
    """
    Uses contextmanager to create and release a file lock based on the
    given path. This allows us to create locks using `with file_lock()`
    to prevent deadlocks related to failure to unlock properly.
    """
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 384)
    fcntl.lockf(lock_fd, fcntl.LOCK_EX)
    yield
    fcntl.lockf(lock_fd, fcntl.LOCK_UN)
    os.close(lock_fd)