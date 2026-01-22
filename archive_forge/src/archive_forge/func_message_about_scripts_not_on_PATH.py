import collections
import compileall
import contextlib
import csv
import importlib
import logging
import os.path
import re
import shutil
import sys
import warnings
from base64 import urlsafe_b64encode
from email.message import Message
from itertools import chain, filterfalse, starmap
from typing import (
from zipfile import ZipFile, ZipInfo
from pip._vendor.distlib.scripts import ScriptMaker
from pip._vendor.distlib.util import get_export_entry
from pip._vendor.packaging.utils import canonicalize_name
from pip._internal.exceptions import InstallationError
from pip._internal.locations import get_major_minor_version
from pip._internal.metadata import (
from pip._internal.models.direct_url import DIRECT_URL_METADATA_NAME, DirectUrl
from pip._internal.models.scheme import SCHEME_KEYS, Scheme
from pip._internal.utils.filesystem import adjacent_tmp_file, replace
from pip._internal.utils.misc import captured_stdout, ensure_dir, hash_file, partition
from pip._internal.utils.unpacking import (
from pip._internal.utils.wheel import parse_wheel
def message_about_scripts_not_on_PATH(scripts: Sequence[str]) -> Optional[str]:
    """Determine if any scripts are not on PATH and format a warning.
    Returns a warning message if one or more scripts are not on PATH,
    otherwise None.
    """
    if not scripts:
        return None
    grouped_by_dir: Dict[str, Set[str]] = collections.defaultdict(set)
    for destfile in scripts:
        parent_dir = os.path.dirname(destfile)
        script_name = os.path.basename(destfile)
        grouped_by_dir[parent_dir].add(script_name)
    not_warn_dirs = [os.path.normcase(os.path.normpath(i)).rstrip(os.sep) for i in os.environ.get('PATH', '').split(os.pathsep)]
    not_warn_dirs.append(os.path.normcase(os.path.normpath(os.path.dirname(sys.executable))))
    warn_for: Dict[str, Set[str]] = {parent_dir: scripts for parent_dir, scripts in grouped_by_dir.items() if os.path.normcase(os.path.normpath(parent_dir)) not in not_warn_dirs}
    if not warn_for:
        return None
    msg_lines = []
    for parent_dir, dir_scripts in warn_for.items():
        sorted_scripts: List[str] = sorted(dir_scripts)
        if len(sorted_scripts) == 1:
            start_text = f'script {sorted_scripts[0]} is'
        else:
            start_text = 'scripts {} are'.format(', '.join(sorted_scripts[:-1]) + ' and ' + sorted_scripts[-1])
        msg_lines.append(f"The {start_text} installed in '{parent_dir}' which is not on PATH.")
    last_line_fmt = 'Consider adding {} to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.'
    if len(msg_lines) == 1:
        msg_lines.append(last_line_fmt.format('this directory'))
    else:
        msg_lines.append(last_line_fmt.format('these directories'))
    warn_for_tilde = any((i[0] == '~' for i in os.environ.get('PATH', '').split(os.pathsep) if i))
    if warn_for_tilde:
        tilde_warning_msg = 'NOTE: The current PATH contains path(s) starting with `~`, which may not be expanded by all applications.'
        msg_lines.append(tilde_warning_msg)
    return '\n'.join(msg_lines)