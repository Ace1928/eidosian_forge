from . import win32
def get_osfhandle(_):
    raise OSError("This isn't windows!")