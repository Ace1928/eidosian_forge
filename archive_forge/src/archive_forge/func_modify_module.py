from __future__ import (absolute_import, division, print_function)
import ast
import base64
import datetime
import json
import os
import shlex
import time
import zipfile
import re
import pkgutil
from ast import AST, Import, ImportFrom
from io import BytesIO
from ansible.release import __version__, __author__
from ansible import constants as C
from ansible.errors import AnsibleError
from ansible.executor.interpreter_discovery import InterpreterDiscoveryRequiredError
from ansible.executor.powershell import module_manifest as ps_manifest
from ansible.module_utils.common.json import AnsibleJSONEncoder
from ansible.module_utils.common.text.converters import to_bytes, to_text, to_native
from ansible.plugins.loader import module_utils_loader
from ansible.utils.collection_loader._collection_finder import _get_collection_metadata, _nested_dict_get
from ansible.executor import action_write_locks
from ansible.utils.display import Display
from collections import namedtuple
import importlib.util
import importlib.machinery
import sys
import {1} as mod
def modify_module(module_name, module_path, module_args, templar, task_vars=None, module_compression='ZIP_STORED', async_timeout=0, become=False, become_method=None, become_user=None, become_password=None, become_flags=None, environment=None, remote_is_local=False):
    """
    Used to insert chunks of code into modules before transfer rather than
    doing regular python imports.  This allows for more efficient transfer in
    a non-bootstrapping scenario by not moving extra files over the wire and
    also takes care of embedding arguments in the transferred modules.

    This version is done in such a way that local imports can still be
    used in the module code, so IDEs don't have to be aware of what is going on.

    Example:

    from ansible.module_utils.basic import *

       ... will result in the insertion of basic.py into the module
       from the module_utils/ directory in the source tree.

    For powershell, this code effectively no-ops, as the exec wrapper requires access to a number of
    properties not available here.

    """
    task_vars = {} if task_vars is None else task_vars
    environment = {} if environment is None else environment
    with open(module_path, 'rb') as f:
        b_module_data = f.read()
    b_module_data, module_style, shebang = _find_module_utils(module_name, b_module_data, module_path, module_args, task_vars, templar, module_compression, async_timeout=async_timeout, become=become, become_method=become_method, become_user=become_user, become_password=become_password, become_flags=become_flags, environment=environment, remote_is_local=remote_is_local)
    if module_style == 'binary':
        return (b_module_data, module_style, to_text(shebang, nonstring='passthru'))
    elif shebang is None:
        interpreter, args = _extract_interpreter(b_module_data)
        if interpreter is not None:
            shebang, new_interpreter = _get_shebang(interpreter, task_vars, templar, args, remote_is_local=remote_is_local)
            b_lines = b_module_data.split(b'\n', 1)
            if interpreter != new_interpreter:
                b_lines[0] = to_bytes(shebang, errors='surrogate_or_strict', nonstring='passthru')
            if os.path.basename(interpreter).startswith(u'python'):
                b_lines.insert(1, b_ENCODING_STRING)
            b_module_data = b'\n'.join(b_lines)
    return (b_module_data, module_style, shebang)