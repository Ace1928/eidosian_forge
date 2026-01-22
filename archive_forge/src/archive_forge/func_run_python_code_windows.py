import ctypes
import os
import struct
import subprocess
import sys
import time
from contextlib import contextmanager
import platform
import traceback
import os, time, sys
def run_python_code_windows(pid, python_code, connect_debugger_tracing=False, show_debug_info=0):
    assert "'" not in python_code, 'Having a single quote messes with our command.'
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=ImportWarning)
        from winappdbg.process import Process
    if not isinstance(python_code, bytes):
        python_code = python_code.encode('utf-8')
    process = Process(pid)
    bits = process.get_bits()
    is_target_process_64 = bits == 64
    with _acquire_mutex('_pydevd_pid_attach_mutex_%s' % (pid,), 10):
        print('--- Connecting to %s bits target (current process is: %s) ---' % (bits, 64 if is_python_64bit() else 32))
        sys.stdout.flush()
        with _win_write_to_shared_named_memory(python_code, pid):
            target_executable = get_target_filename(is_target_process_64, 'inject_dll_', '.exe')
            if not target_executable:
                raise RuntimeError('Could not find expected .exe file to inject dll in attach to process.')
            target_dll = get_target_filename(is_target_process_64)
            if not target_dll:
                raise RuntimeError('Could not find expected .dll file in attach to process.')
            print('\n--- Injecting attach dll: %s into pid: %s ---' % (os.path.basename(target_dll), pid))
            sys.stdout.flush()
            args = [target_executable, str(pid), target_dll]
            subprocess.check_call(args)
            target_dll_run_on_dllmain = get_target_filename(is_target_process_64, 'run_code_on_dllmain_', '.dll')
            if not target_dll_run_on_dllmain:
                raise RuntimeError('Could not find expected .dll in attach to process.')
            with _create_win_event('_pydevd_pid_event_%s' % (pid,)) as event:
                print('\n--- Injecting run code dll: %s into pid: %s ---' % (os.path.basename(target_dll_run_on_dllmain), pid))
                sys.stdout.flush()
                args = [target_executable, str(pid), target_dll_run_on_dllmain]
                subprocess.check_call(args)
                if not event.wait_for_event_set(15):
                    print('Timeout error: the attach may not have completed.')
                    sys.stdout.flush()
            print('--- Finished dll injection ---\n')
            sys.stdout.flush()
    return 0