def async_raise(tid, exctype):
    """raises the exception, performs cleanup if needed.

    tid is the value given by thread.get_ident() (an integer).
    Raise SystemExit to kill a thread."""
    if not isinstance(exctype, type):
        raise TypeError('Only types can be raised (not instances)')
    if not isinstance(tid, int):
        raise TypeError('tid must be an integer')
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), ctypes.py_object(exctype))
    if res == 0:
        raise ValueError('invalid thread id')
    elif res != 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), 0)
        raise SystemError('PyThreadState_SetAsyncExc failed')