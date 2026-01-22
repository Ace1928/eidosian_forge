import sys
import time
import signal
import OpenGL.GLUT as glut
import OpenGL.platform as platform
from timeit import default_timer as clock
def glut_int_handler(signum, frame):
    signal.signal(signal.SIGINT, signal.default_int_handler)
    print('\nKeyboardInterrupt')