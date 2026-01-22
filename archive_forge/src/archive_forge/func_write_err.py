from _pydev_bundle.pydev_imports import execfile
from _pydevd_bundle import pydevd_dont_trace
import types
from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_constants import get_global_debugger
def write_err(*args):
    py_db = get_global_debugger()
    if py_db is not None:
        new_lst = []
        for a in args:
            new_lst.append(str(a))
        msg = ' '.join(new_lst)
        s = 'code reload: %s\n' % (msg,)
        cmd = py_db.cmd_factory.make_io_message(s, 2)
        if py_db.writer is not None:
            py_db.writer.add_command(cmd)