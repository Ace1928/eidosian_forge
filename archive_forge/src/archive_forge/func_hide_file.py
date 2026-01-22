import platform
@windows_only
def hide_file(path):
    """
    Set the hidden attribute on a file or directory.

    From http://stackoverflow.com/questions/19622133/

    `path` must be text.
    """
    import ctypes
    __import__('ctypes.wintypes')
    SetFileAttributes = ctypes.windll.kernel32.SetFileAttributesW
    SetFileAttributes.argtypes = (ctypes.wintypes.LPWSTR, ctypes.wintypes.DWORD)
    SetFileAttributes.restype = ctypes.wintypes.BOOL
    FILE_ATTRIBUTE_HIDDEN = 2
    ret = SetFileAttributes(path, FILE_ATTRIBUTE_HIDDEN)
    if not ret:
        raise ctypes.WinError()