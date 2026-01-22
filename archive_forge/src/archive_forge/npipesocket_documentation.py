import functools
import time
import io
import win32file
import win32pipe
import pywintypes
import win32event
import win32api
 Partial implementation of the socket API over windows named pipes.
        This implementation is only designed to be used as a client socket,
        and server-specific methods (bind, listen, accept...) are not
        implemented.
    