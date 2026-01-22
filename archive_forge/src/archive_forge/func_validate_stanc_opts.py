import io
import json
import os
import platform
import shutil
import subprocess
from copy import copy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union
from cmdstanpy.utils import get_logger
from cmdstanpy.utils.cmdstan import (
from cmdstanpy.utils.command import do_command
from cmdstanpy.utils.filesystem import SanitizedOrTmpFilePath
def validate_stanc_opts(self) -> None:
    """
        Check stanc compiler args and consistency between stanc and C++ options.
        Raise ValueError if bad config is found.
        """
    if self._stanc_options is None:
        return
    ignore = []
    paths = None
    has_o_flag = False
    for deprecated, replacement in STANC_DEPRECATED_OPTS.items():
        if deprecated in self._stanc_options:
            if replacement:
                get_logger().warning('compiler option "%s" is deprecated, use "%s" instead', deprecated, replacement)
                self._stanc_options[replacement] = copy(self._stanc_options[deprecated])
                del self._stanc_options[deprecated]
            else:
                get_logger().warning('compiler option "%s" is deprecated and should not be used', deprecated)
    for key, val in self._stanc_options.items():
        if key in STANC_IGNORE_OPTS:
            get_logger().info('ignoring compiler option: %s', key)
            ignore.append(key)
        elif key not in STANC_OPTS:
            raise ValueError(f'unknown stanc compiler option: {key}')
        elif key == 'include-paths':
            paths = val
            if isinstance(val, str):
                paths = val.split(',')
            elif not isinstance(val, list):
                raise ValueError(f'Invalid include-paths, expecting list or string, found type: {type(val)}.')
        elif key == 'use-opencl':
            if self._cpp_options is None:
                self._cpp_options = {'STAN_OPENCL': 'TRUE'}
            else:
                self._cpp_options['STAN_OPENCL'] = 'TRUE'
        elif key.startswith('O'):
            if has_o_flag:
                get_logger().warning('More than one of (O, O1, O2, Oexperimental)optimizations passed. Only the last one willbe used')
            else:
                has_o_flag = True
    for opt in ignore:
        del self._stanc_options[opt]
    if paths is not None:
        bad_paths = [dir for dir in paths if not os.path.exists(dir)]
        if any(bad_paths):
            raise ValueError('invalid include paths: {}'.format(', '.join(bad_paths)))
        self._stanc_options['include-paths'] = [os.path.abspath(os.path.expanduser(path)) for path in paths]