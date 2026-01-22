from __future__ import (absolute_import, division, print_function)
import atexit
import configparser
import os
import os.path
import sys
import stat
import tempfile
from collections import namedtuple
from collections.abc import Mapping, Sequence
from jinja2.nativetypes import NativeEnvironment
from ansible.errors import AnsibleOptionsError, AnsibleError
from ansible.module_utils.common.text.converters import to_text, to_bytes, to_native
from ansible.module_utils.common.yaml import yaml_load
from ansible.module_utils.six import string_types
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.parsing.quoting import unquote
from ansible.parsing.yaml.objects import AnsibleVaultEncryptedUnicode
from ansible.utils import py3compat
from ansible.utils.path import cleanup_tmp_file, makedirs_safe, unfrackpath
def find_ini_config_file(warnings=None):
    """ Load INI Config File order(first found is used): ENV, CWD, HOME, /etc/ansible """
    if warnings is None:
        warnings = set()
    SENTINEL = object
    potential_paths = []
    path_from_env = os.getenv('ANSIBLE_CONFIG', SENTINEL)
    if path_from_env is not SENTINEL:
        path_from_env = unfrackpath(path_from_env, follow=False)
        if os.path.isdir(to_bytes(path_from_env)):
            path_from_env = os.path.join(path_from_env, 'ansible.cfg')
        potential_paths.append(path_from_env)
    warn_cmd_public = False
    try:
        cwd = os.getcwd()
        perms = os.stat(cwd)
        cwd_cfg = os.path.join(cwd, 'ansible.cfg')
        if perms.st_mode & stat.S_IWOTH:
            if os.path.exists(cwd_cfg):
                warn_cmd_public = True
        else:
            potential_paths.append(to_text(cwd_cfg, errors='surrogate_or_strict'))
    except OSError:
        pass
    potential_paths.append(unfrackpath('~/.ansible.cfg', follow=False))
    potential_paths.append('/etc/ansible/ansible.cfg')
    for path in potential_paths:
        b_path = to_bytes(path)
        if os.path.exists(b_path) and os.access(b_path, os.R_OK):
            break
    else:
        path = None
    if path_from_env != path and warn_cmd_public:
        warnings.add(u'Ansible is being run in a world writable directory (%s), ignoring it as an ansible.cfg source. For more information see https://docs.ansible.com/ansible/devel/reference_appendices/config.html#cfg-in-world-writable-dir' % to_text(cwd))
    return path