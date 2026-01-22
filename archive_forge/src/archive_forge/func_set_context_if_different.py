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
def set_context_if_different(self, path, context, changed, diff=None):
    if not self.selinux_enabled():
        return changed
    if self.check_file_absent_if_check_mode(path):
        return True
    cur_context = self.selinux_context(path)
    new_context = list(cur_context)
    is_special_se, sp_context = self.is_special_selinux_path(path)
    if is_special_se:
        new_context = sp_context
    else:
        for i in range(len(cur_context)):
            if len(context) > i:
                if context[i] is not None and context[i] != cur_context[i]:
                    new_context[i] = context[i]
                elif context[i] is None:
                    new_context[i] = cur_context[i]
    if cur_context != new_context:
        if diff is not None:
            if 'before' not in diff:
                diff['before'] = {}
            diff['before']['secontext'] = cur_context
            if 'after' not in diff:
                diff['after'] = {}
            diff['after']['secontext'] = new_context
        try:
            if self.check_mode:
                return True
            rc = selinux.lsetfilecon(to_native(path), ':'.join(new_context))
        except OSError as e:
            self.fail_json(path=path, msg='invalid selinux context: %s' % to_native(e), new_context=new_context, cur_context=cur_context, input_was=context)
        if rc != 0:
            self.fail_json(path=path, msg='set selinux context failed')
        changed = True
    return changed