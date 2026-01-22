from functools import wraps
def getThreadID():
    if threadingmodule is None:
        return _dummyID
    return threadingmodule.current_thread().ident