import contextlib
import os
import re
import sys
from distutils.core import Command
from distutils.errors import *
from distutils.sysconfig import customize_compiler, get_python_version
from distutils.sysconfig import get_config_h_filename
from distutils.dep_util import newer_group
from distutils.extension import Extension
from distutils.util import get_platform
from distutils import log
from site import USER_BASE
def check_extensions_list(self, extensions):
    """Ensure that the list of extensions (presumably provided as a
        command option 'extensions') is valid, i.e. it is a list of
        Extension objects.  We also support the old-style list of 2-tuples,
        where the tuples are (ext_name, build_info), which are converted to
        Extension instances here.

        Raise DistutilsSetupError if the structure is invalid anywhere;
        just returns otherwise.
        """
    if not isinstance(extensions, list):
        raise DistutilsSetupError("'ext_modules' option must be a list of Extension instances")
    for i, ext in enumerate(extensions):
        if isinstance(ext, Extension):
            continue
        if not isinstance(ext, tuple) or len(ext) != 2:
            raise DistutilsSetupError("each element of 'ext_modules' option must be an Extension instance or 2-tuple")
        ext_name, build_info = ext
        log.warn("old-style (ext_name, build_info) tuple found in ext_modules for extension '%s' -- please convert to Extension instance", ext_name)
        if not (isinstance(ext_name, str) and extension_name_re.match(ext_name)):
            raise DistutilsSetupError("first element of each tuple in 'ext_modules' must be the extension name (a string)")
        if not isinstance(build_info, dict):
            raise DistutilsSetupError("second element of each tuple in 'ext_modules' must be a dictionary (build info)")
        ext = Extension(ext_name, build_info['sources'])
        for key in ('include_dirs', 'library_dirs', 'libraries', 'extra_objects', 'extra_compile_args', 'extra_link_args'):
            val = build_info.get(key)
            if val is not None:
                setattr(ext, key, val)
        ext.runtime_library_dirs = build_info.get('rpath')
        if 'def_file' in build_info:
            log.warn("'def_file' element of build info dict no longer supported")
        macros = build_info.get('macros')
        if macros:
            ext.define_macros = []
            ext.undef_macros = []
            for macro in macros:
                if not (isinstance(macro, tuple) and len(macro) in (1, 2)):
                    raise DistutilsSetupError("'macros' element of build info dict must be 1- or 2-tuple")
                if len(macro) == 1:
                    ext.undef_macros.append(macro[0])
                elif len(macro) == 2:
                    ext.define_macros.append(macro)
        extensions[i] = ext