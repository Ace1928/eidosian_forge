import sys
import os
import re
import copy
import warnings
import subprocess
import textwrap
from glob import glob
from functools import reduce
from configparser import NoOptionError
from configparser import RawConfigParser as ConfigParser
from distutils.errors import DistutilsError
from distutils.dist import Distribution
import sysconfig
from numpy.distutils import log
from distutils.util import get_platform
from numpy.distutils.exec_command import (
from numpy.distutils.misc_util import (is_sequence, is_string,
from numpy.distutils.command.config import config as cmd_config
from numpy.distutils import customized_ccompiler as _customized_ccompiler
from numpy.distutils import _shell_utils
import distutils.ccompiler
import tempfile
import shutil
import platform
def get_atlas_version(**config):
    libraries = config.get('libraries', [])
    library_dirs = config.get('library_dirs', [])
    key = (tuple(libraries), tuple(library_dirs))
    if key in _cached_atlas_version:
        return _cached_atlas_version[key]
    c = cmd_config(Distribution())
    atlas_version = None
    info = {}
    try:
        s, o = c.get_output(atlas_version_c_text, libraries=libraries, library_dirs=library_dirs)
        if s and re.search('undefined reference to `_gfortran', o, re.M):
            s, o = c.get_output(atlas_version_c_text, libraries=libraries + ['gfortran'], library_dirs=library_dirs)
            if not s:
                warnings.warn(textwrap.dedent('\n                    *****************************************************\n                    Linkage with ATLAS requires gfortran. Use\n\n                      python setup.py config_fc --fcompiler=gnu95 ...\n\n                    when building extension libraries that use ATLAS.\n                    Make sure that -lgfortran is used for C++ extensions.\n                    *****************************************************\n                    '), stacklevel=2)
                dict_append(info, language='f90', define_macros=[('ATLAS_REQUIRES_GFORTRAN', None)])
    except Exception:
        for o in library_dirs:
            m = re.search('ATLAS_(?P<version>\\d+[.]\\d+[.]\\d+)_', o)
            if m:
                atlas_version = m.group('version')
            if atlas_version is not None:
                break
        if atlas_version is None:
            atlas_version = os.environ.get('ATLAS_VERSION', None)
        if atlas_version:
            dict_append(info, define_macros=[('ATLAS_INFO', _c_string_literal(atlas_version))])
        else:
            dict_append(info, define_macros=[('NO_ATLAS_INFO', -1)])
        return (atlas_version or '?.?.?', info)
    if not s:
        m = re.search('ATLAS version (?P<version>\\d+[.]\\d+[.]\\d+)', o)
        if m:
            atlas_version = m.group('version')
    if atlas_version is None:
        if re.search('undefined symbol: ATL_buildinfo', o, re.M):
            atlas_version = '3.2.1_pre3.3.6'
        else:
            log.info('Status: %d', s)
            log.info('Output: %s', o)
    elif atlas_version == '3.2.1_pre3.3.6':
        dict_append(info, define_macros=[('NO_ATLAS_INFO', -2)])
    else:
        dict_append(info, define_macros=[('ATLAS_INFO', _c_string_literal(atlas_version))])
    result = _cached_atlas_version[key] = (atlas_version, info)
    return result