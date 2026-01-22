import os, glob, re, sys
from distutils import sysconfig
def python_link_data():
    libdir = sysconfig.get_config_var('LIBDIR')
    if libdir is None:
        libdir = os.path.abspath(os.path.join(sysconfig.get_config_var('LIBDEST'), '..', 'libs'))
    version = python_version()
    version_no_dots = version.replace('.', '')
    flags = {}
    flags['libdir'] = libdir
    if sys.platform == 'win32':
        suffix = '_d' if is_debug() else ''
        flags['lib'] = 'python{}{}'.format(version_no_dots, suffix)
    elif sys.platform == 'darwin':
        flags['lib'] = 'python{}'.format(version)
    elif sys.version_info[0] < 3:
        suffix = '_d' if is_debug() else ''
        flags['lib'] = 'python{}{}'.format(version, suffix)
    else:
        flags['lib'] = 'python{}{}'.format(version, sys.abiflags)
    return flags