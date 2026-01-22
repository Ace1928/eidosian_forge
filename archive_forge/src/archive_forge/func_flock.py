import errno
import msvcrt
import pywintypes
import win32con
import win32file
def flock(fd, flags):
    lockf(fd, flags, 4294901760, 0, 0)