from . import win32
def enable_vt_processing(fd):
    if win32.windll is None or not win32.winapi_test():
        return False
    try:
        handle = get_osfhandle(fd)
        mode = win32.GetConsoleMode(handle)
        win32.SetConsoleMode(handle, mode | win32.ENABLE_VIRTUAL_TERMINAL_PROCESSING)
        mode = win32.GetConsoleMode(handle)
        if mode & win32.ENABLE_VIRTUAL_TERMINAL_PROCESSING:
            return True
    except (OSError, TypeError):
        return False