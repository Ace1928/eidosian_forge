from __future__ import annotations
import collections
import contextlib
import os
import platform
import pstats
import re
import sys
from . import config
from .util import gc_collect
from ..util import has_compiled_ext
@property
def platform_key(self):
    dbapi_key = config.db.name + '_' + config.db.driver
    if config.db.name == 'sqlite' and config.db.dialect._is_url_file_db(config.db.url):
        dbapi_key += '_file'
    py_version = '.'.join([str(v) for v in sys.version_info[0:2]])
    platform_tokens = [platform.machine(), platform.system().lower(), platform.python_implementation().lower(), py_version, dbapi_key]
    platform_tokens.append('dbapiunicode')
    _has_cext = has_compiled_ext()
    platform_tokens.append(_has_cext and 'cextensions' or 'nocextensions')
    return '_'.join(platform_tokens)