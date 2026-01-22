import sys
def connect_named_pipe(pipe_handle, overlapped=None):
    try:
        error = win32pipe.ConnectNamedPipe(pipe_handle, overlapped)
        return error
    except pywintypes.error as e:
        return e.winerror