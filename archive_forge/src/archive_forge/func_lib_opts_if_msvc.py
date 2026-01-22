import os
from distutils.msvccompiler import MSVCCompiler as _MSVCCompiler
from .system_info import platform_bits
def lib_opts_if_msvc(build_cmd):
    """ Add flags if we are using MSVC compiler

    We can't see `build_cmd` in our scope, because we have not initialized
    the distutils build command, so use this deferred calculation to run
    when we are building the library.
    """
    if build_cmd.compiler.compiler_type != 'msvc':
        return []
    flags = ['/GL-']
    if build_cmd.compiler_opt.cc_test_flags(['-d2VolatileMetadata-']):
        flags.append('-d2VolatileMetadata-')
    return flags