from collections import namedtuple
import os
import re
import subprocess
from textwrap import dedent
from numba import config
def collect_gdbinfo():
    """Prints information to stdout about the gdb setup that Numba has found"""
    gdb_state = None
    gdb_has_python = False
    gdb_has_numpy = False
    gdb_python_version = 'No Python support'
    gdb_python_numpy_version = 'No NumPy support'
    try:
        gdb_wrapper = _GDBTestWrapper()
        status = gdb_wrapper.check_launch()
        if not gdb_wrapper.success(status):
            msg = f"gdb at '{gdb_wrapper.gdb_binary}' does not appear to work.\nstdout: {status.stdout}\nstderr: {status.stderr}"
            raise ValueError(msg)
        gdb_state = gdb_wrapper.gdb_binary
    except Exception as e:
        gdb_state = f'Testing gdb binary failed. Reported Error: {e}'
    else:
        status = gdb_wrapper.check_python()
        if gdb_wrapper.success(status):
            version_match = re.match('\\((\\d+),\\s+(\\d+)\\)', status.stdout.strip())
            if version_match is not None:
                pymajor, pyminor = version_match.groups()
                gdb_python_version = f'{pymajor}.{pyminor}'
                gdb_has_python = True
                status = gdb_wrapper.check_numpy()
                if gdb_wrapper.success(status):
                    if 'Traceback' not in status.stderr.strip():
                        if status.stdout.strip() == 'True':
                            gdb_has_numpy = True
                            gdb_python_numpy_version = 'Unknown'
                            status = gdb_wrapper.check_numpy_version()
                            if gdb_wrapper.success(status):
                                if 'Traceback' not in status.stderr.strip():
                                    gdb_python_numpy_version = status.stdout.strip()
    if gdb_has_python:
        if gdb_has_numpy:
            print_ext_supported = 'Full (Python and NumPy supported)'
        else:
            print_ext_supported = 'Partial (Python only, no NumPy support)'
    else:
        print_ext_supported = 'None'
    print_ext_file = 'gdb_print_extension.py'
    print_ext_path = os.path.join(os.path.dirname(__file__), print_ext_file)
    return _gdb_info(gdb_state, print_ext_path, gdb_python_version, gdb_python_numpy_version, print_ext_supported)