import locale
import logging
import os
import pprint
import re
import sys
import warnings
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any, Dict
from typing_extensions import Literal
import numpy as np
def read_rcfile(fname):
    """Return :class:`arviz.RcParams` from the contents of the given file.

    Unlike `rc_params_from_file`, the configuration class only contains the
    parameters specified in the file (i.e. default values are not filled in).
    """
    _error_details_fmt = 'line #%d\n\t"%s"\n\tin file "%s"'
    config = RcParams()
    with open(fname, 'r', encoding='utf8') as rcfile:
        try:
            multiline = False
            for line_no, line in enumerate(rcfile, 1):
                strippedline = line.split('#', 1)[0].strip()
                if not strippedline:
                    continue
                if multiline:
                    if strippedline == '}':
                        multiline = False
                        val = aux_val
                    else:
                        aux_val.append(strippedline)
                        continue
                else:
                    tup = strippedline.split(':', 1)
                    if len(tup) != 2:
                        error_details = _error_details_fmt % (line_no, line, fname)
                        _log.warning('Illegal %s', error_details)
                        continue
                    key, val = tup
                    key = key.strip()
                    val = val.strip()
                    if key in config:
                        _log.warning('Duplicate key in file %r line #%d.', fname, line_no)
                    if key in {'data.metagroups'}:
                        aux_val = []
                        multiline = True
                        continue
                try:
                    config[key] = val
                except ValueError as verr:
                    error_details = _error_details_fmt % (line_no, line, fname)
                    raise ValueError(f'Bad val {val} on {error_details}\n\t{str(verr)}') from verr
        except UnicodeDecodeError:
            _log.warning('Cannot decode configuration file %s with encoding %s, check LANG and LC_* variables.', fname, locale.getpreferredencoding(do_setlocale=False) or 'utf-8 (default)')
            raise
        return config