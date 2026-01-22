import json
import os
import re
import sys
from importlib.util import find_spec
import pydevd
from _pydevd_bundle import pydevd_runpy as runpy
import debugpy
from debugpy.common import log
from debugpy.server import api
import codecs;
import json;
import sys;
import attach_pid_injected;
def attach_to_pid():
    pid = options.target
    log.info('Attaching to process with PID={0}', pid)
    encode = lambda s: list(bytearray(s.encode('utf-8'))) if s is not None else None
    script_dir = os.path.dirname(debugpy.server.__file__)
    assert os.path.exists(script_dir)
    script_dir = encode(script_dir)
    setup = {'mode': options.mode, 'address': options.address, 'wait_for_client': options.wait_for_client, 'log_to': options.log_to, 'adapter_access_token': options.adapter_access_token}
    setup = encode(json.dumps(setup))
    python_code = '\nimport codecs;\nimport json;\nimport sys;\n\ndecode = lambda s: codecs.utf_8_decode(bytearray(s))[0] if s is not None else None;\n\nscript_dir = decode({script_dir});\nsetup = json.loads(decode({setup}));\n\nsys.path.insert(0, script_dir);\nimport attach_pid_injected;\ndel sys.path[0];\n\nattach_pid_injected.attach(setup);\n'
    python_code = python_code.replace('\r', '').replace('\n', '').format(script_dir=script_dir, setup=setup)
    log.info('Code to be injected: \n{0}', python_code.replace(';', ';\n'))
    assert not {'"', "'", '\r', '\n'} & set(python_code), 'Injected code should not contain any single quotes, double quotes, or newlines.'
    pydevd_attach_to_process_path = os.path.join(os.path.dirname(pydevd.__file__), 'pydevd_attach_to_process')
    assert os.path.exists(pydevd_attach_to_process_path)
    sys.path.append(pydevd_attach_to_process_path)
    try:
        import add_code_to_python_process
        log.info('Injecting code into process with PID={0} ...', pid)
        add_code_to_python_process.run_python_code(pid, python_code, connect_debugger_tracing=True, show_debug_info=int(os.getenv('DEBUGPY_ATTACH_BY_PID_DEBUG_INFO', '0')))
    except Exception:
        log.reraise_exception('Code injection into PID={0} failed:', pid)
    log.info('Code injection into PID={0} completed.', pid)