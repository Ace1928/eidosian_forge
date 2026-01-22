from __future__ import absolute_import, division, print_function
import sys
import __main__
import atexit
import errno
import datetime
import grp
import fcntl
import locale
import os
import pwd
import platform
import re
import select
import shlex
import shutil
import signal
import stat
import subprocess
import tempfile
import time
import traceback
import types
from itertools import chain, repeat
from ansible.module_utils.compat import selectors
from ._text import to_native, to_bytes, to_text
from ansible.module_utils.common.text.converters import (
from ansible.module_utils.common.arg_spec import ModuleArgumentSpecValidator
from ansible.module_utils.common.text.formatters import (
import hashlib
from ansible.module_utils.six.moves.collections_abc import (
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.common.file import (
from ansible.module_utils.common.sys_info import (
from ansible.module_utils.pycompat24 import get_exception, literal_eval
from ansible.module_utils.common.parameters import (
from ansible.module_utils.errors import AnsibleFallbackNotFound, AnsibleValidationErrorMultiple, UnsupportedError
from ansible.module_utils.six import (
from ansible.module_utils.six.moves import map, reduce, shlex_quote
from ansible.module_utils.common.validation import (
from ansible.module_utils.common._utils import get_all_subclasses as _get_all_subclasses
from ansible.module_utils.parsing.convert_bool import BOOLEANS, BOOLEANS_FALSE, BOOLEANS_TRUE, boolean
from ansible.module_utils.common.warnings import (
def load_file_common_arguments(self, params, path=None):
    """
        many modules deal with files, this encapsulates common
        options that the file module accepts such that it is directly
        available to all modules and they can share code.

        Allows to overwrite the path/dest module argument by providing path.
        """
    if path is None:
        path = params.get('path', params.get('dest', None))
    if path is None:
        return {}
    else:
        path = os.path.expanduser(os.path.expandvars(path))
    b_path = to_bytes(path, errors='surrogate_or_strict')
    if params.get('follow', False) and os.path.islink(b_path):
        b_path = os.path.realpath(b_path)
        path = to_native(b_path)
    mode = params.get('mode', None)
    owner = params.get('owner', None)
    group = params.get('group', None)
    seuser = params.get('seuser', None)
    serole = params.get('serole', None)
    setype = params.get('setype', None)
    selevel = params.get('selevel', None)
    secontext = [seuser, serole, setype]
    if self.selinux_mls_enabled():
        secontext.append(selevel)
    default_secontext = self.selinux_default_context(path)
    for i in range(len(default_secontext)):
        if i is not None and secontext[i] == '_default':
            secontext[i] = default_secontext[i]
    attributes = params.get('attributes', None)
    return dict(path=path, mode=mode, owner=owner, group=group, seuser=seuser, serole=serole, setype=setype, selevel=selevel, secontext=secontext, attributes=attributes)