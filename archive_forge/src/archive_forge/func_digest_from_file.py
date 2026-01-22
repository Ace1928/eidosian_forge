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
def digest_from_file(self, filename, algorithm):
    """ Return hex digest of local file for a digest_method specified by name, or None if file is not present. """
    b_filename = to_bytes(filename, errors='surrogate_or_strict')
    if not os.path.exists(b_filename):
        return None
    if os.path.isdir(b_filename):
        self.fail_json(msg='attempted to take checksum of directory: %s' % filename)
    if hasattr(algorithm, 'hexdigest'):
        digest_method = algorithm
    else:
        try:
            digest_method = AVAILABLE_HASH_ALGORITHMS[algorithm]()
        except KeyError:
            self.fail_json(msg="Could not hash file '%s' with algorithm '%s'. Available algorithms: %s" % (filename, algorithm, ', '.join(AVAILABLE_HASH_ALGORITHMS)))
    blocksize = 64 * 1024
    infile = open(os.path.realpath(b_filename), 'rb')
    block = infile.read(blocksize)
    while block:
        digest_method.update(block)
        block = infile.read(blocksize)
    infile.close()
    return digest_method.hexdigest()