import sys
def windows_create_pipe(sAttrs=-1, nSize=None):
    if sAttrs == -1:
        sAttrs = win32security.SECURITY_ATTRIBUTES()
        sAttrs.bInheritHandle = 1
    if nSize is None:
        nSize = 0
    try:
        read_pipe, write_pipe = win32pipe.CreatePipe(sAttrs, nSize)
    except pywintypes.error:
        raise
    return (read_pipe, write_pipe)