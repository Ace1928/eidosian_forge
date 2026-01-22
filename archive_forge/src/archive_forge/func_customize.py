import os
import sys
import re
from pathlib import Path
from distutils.sysconfig import get_python_lib
from distutils.fancy_getopt import FancyGetopt
from distutils.errors import DistutilsModuleError, \
from distutils.util import split_quoted, strtobool
from numpy.distutils.ccompiler import CCompiler, gen_lib_options
from numpy.distutils import log
from numpy.distutils.misc_util import is_string, all_strings, is_sequence, \
from numpy.distutils.exec_command import find_executable
from numpy.distutils import _shell_utils
from .environment import EnvironmentConfig
def customize(self, dist=None):
    """Customize Fortran compiler.

        This method gets Fortran compiler specific information from
        (i) class definition, (ii) environment, (iii) distutils config
        files, and (iv) command line (later overrides earlier).

        This method should be always called after constructing a
        compiler instance. But not in __init__ because Distribution
        instance is needed for (iii) and (iv).
        """
    log.info('customize %s' % self.__class__.__name__)
    self._is_customised = True
    self.distutils_vars.use_distribution(dist)
    self.command_vars.use_distribution(dist)
    self.flag_vars.use_distribution(dist)
    self.update_executables()
    self.find_executables()
    noopt = self.distutils_vars.get('noopt', False)
    noarch = self.distutils_vars.get('noarch', noopt)
    debug = self.distutils_vars.get('debug', False)
    f77 = self.command_vars.compiler_f77
    f90 = self.command_vars.compiler_f90
    f77flags = []
    f90flags = []
    freeflags = []
    fixflags = []
    if f77:
        f77 = _shell_utils.NativeParser.split(f77)
        f77flags = self.flag_vars.f77
    if f90:
        f90 = _shell_utils.NativeParser.split(f90)
        f90flags = self.flag_vars.f90
        freeflags = self.flag_vars.free
    fix = self.command_vars.compiler_fix
    if fix:
        fix = _shell_utils.NativeParser.split(fix)
        fixflags = self.flag_vars.fix + f90flags
    oflags, aflags, dflags = ([], [], [])

    def get_flags(tag, flags):
        flags.extend(getattr(self.flag_vars, tag))
        this_get = getattr(self, 'get_flags_' + tag)
        for name, c, flagvar in [('f77', f77, f77flags), ('f90', f90, f90flags), ('f90', fix, fixflags)]:
            t = '%s_%s' % (tag, name)
            if c and this_get is not getattr(self, 'get_flags_' + t):
                flagvar.extend(getattr(self.flag_vars, t))
    if not noopt:
        get_flags('opt', oflags)
        if not noarch:
            get_flags('arch', aflags)
    if debug:
        get_flags('debug', dflags)
    fflags = self.flag_vars.flags + dflags + oflags + aflags
    if f77:
        self.set_commands(compiler_f77=f77 + f77flags + fflags)
    if f90:
        self.set_commands(compiler_f90=f90 + freeflags + f90flags + fflags)
    if fix:
        self.set_commands(compiler_fix=fix + fixflags + fflags)
    linker_so = self.linker_so
    if linker_so:
        linker_so_flags = self.flag_vars.linker_so
        if sys.platform.startswith('aix'):
            python_lib = get_python_lib(standard_lib=1)
            ld_so_aix = os.path.join(python_lib, 'config', 'ld_so_aix')
            python_exp = os.path.join(python_lib, 'config', 'python.exp')
            linker_so = [ld_so_aix] + linker_so + ['-bI:' + python_exp]
        if sys.platform.startswith('os400'):
            from distutils.sysconfig import get_config_var
            python_config = get_config_var('LIBPL')
            ld_so_aix = os.path.join(python_config, 'ld_so_aix')
            python_exp = os.path.join(python_config, 'python.exp')
            linker_so = [ld_so_aix] + linker_so + ['-bI:' + python_exp]
        self.set_commands(linker_so=linker_so + linker_so_flags)
    linker_exe = self.linker_exe
    if linker_exe:
        linker_exe_flags = self.flag_vars.linker_exe
        self.set_commands(linker_exe=linker_exe + linker_exe_flags)
    ar = self.command_vars.archiver
    if ar:
        arflags = self.flag_vars.ar
        self.set_commands(archiver=[ar] + arflags)
    self.set_library_dirs(self.get_library_dirs())
    self.set_libraries(self.get_libraries())