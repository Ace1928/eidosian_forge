from _pydev_bundle import pydev_log
from _pydevd_bundle import pydevd_import_class
from _pydevd_bundle.pydevd_frame_utils import add_exception_to_frame
from _pydev_bundle._pydev_saved_modules import threading
def stop_on_unhandled_exception(py_db, thread, additional_info, arg):
    exctype, value, tb = arg
    break_on_uncaught_exceptions = py_db.break_on_uncaught_exceptions
    if break_on_uncaught_exceptions:
        exception_breakpoint = py_db.get_exception_breakpoint(exctype, break_on_uncaught_exceptions)
    else:
        exception_breakpoint = None
    if not exception_breakpoint:
        return
    if tb is None:
        return
    if exctype is KeyboardInterrupt:
        return
    if exctype is SystemExit and py_db.ignore_system_exit_code(value):
        return
    frames = []
    user_frame = None
    while tb is not None:
        if not py_db.exclude_exception_by_filter(exception_breakpoint, tb):
            user_frame = tb.tb_frame
        frames.append(tb.tb_frame)
        tb = tb.tb_next
    if user_frame is None:
        return
    frames_byid = dict([(id(frame), frame) for frame in frames])
    add_exception_to_frame(user_frame, arg)
    if exception_breakpoint.condition is not None:
        eval_result = py_db.handle_breakpoint_condition(additional_info, exception_breakpoint, user_frame)
        if not eval_result:
            return
    if exception_breakpoint.expression is not None:
        py_db.handle_breakpoint_expression(exception_breakpoint, additional_info, user_frame)
    try:
        additional_info.pydev_message = exception_breakpoint.qname
    except:
        additional_info.pydev_message = exception_breakpoint.qname.encode('utf-8')
    pydev_log.debug('Handling post-mortem stop on exception breakpoint %s' % (exception_breakpoint.qname,))
    py_db.do_stop_on_unhandled_exception(thread, user_frame, frames_byid, arg)