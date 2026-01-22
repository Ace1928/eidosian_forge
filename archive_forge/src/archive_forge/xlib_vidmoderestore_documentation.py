import ctypes
import os
import signal
import struct
import threading
from pyglet.libs.x11 import xlib
from pyglet.util import asbytes
Fork a child process and inform it of mode changes to each screen.  The
child waits until the parent process dies, and then connects to each X server 
with a mode change and restores the mode.

This emulates the behaviour of Windows and Mac, so that resolution changes
made by an application are not permanent after the program exits, even if
the process is terminated uncleanly.

The child process is communicated to via a pipe, and watches for parent
death with a Linux extension signal handler.
