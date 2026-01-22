import importlib
import os
import subprocess
import sys
import sysconfig
from pathlib import Path
import PySide6 as ref_mod
def designer():
    init_virtual_env()
    major_version = sys.version_info[0]
    minor_version = sys.version_info[1]
    os.environ['PY_MAJOR_VERSION'] = str(major_version)
    os.environ['PY_MINOR_VERSION'] = str(minor_version)
    if sys.platform == 'linux':
        version = f'{major_version}.{minor_version}'
        library_name = f'libpython{version}{sys.abiflags}.so'
        if is_pyenv_python():
            library_name = str(Path(sysconfig.get_config_var('LIBDIR')) / library_name)
        os.environ['LD_PRELOAD'] = library_name
    elif sys.platform == 'darwin':
        library_name = sysconfig.get_config_var('LDLIBRARY')
        framework_prefix = sysconfig.get_config_var('PYTHONFRAMEWORKPREFIX')
        lib_path = None
        if framework_prefix:
            lib_path = os.fspath(Path(framework_prefix) / library_name)
        elif is_pyenv_python():
            lib_path = str(Path(sysconfig.get_config_var('LIBDIR')) / library_name)
        else:
            print('Unable to find Python library directory. Use a framework build of Python.', file=sys.stderr)
            sys.exit(0)
        os.environ['DYLD_INSERT_LIBRARIES'] = lib_path
    elif sys.platform == 'win32':
        if is_virtual_env():
            _extend_path_var('PATH', os.fspath(Path(sys._base_executable).parent), True)
    qt_tool_wrapper(ui_tool_binary('designer'), sys.argv[1:])