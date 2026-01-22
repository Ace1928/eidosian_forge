import atexit
import os
import signal
import sys
import time
def reap_process_group(*args):

    def sigterm_handler(*args):
        time.sleep(SIGTERM_GRACE_PERIOD_SECONDS)
        if sys.platform == 'win32':
            atexit.unregister(sigterm_handler)
            os.kill(0, signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(0, signal.SIGKILL)
    if sys.platform == 'win32':
        atexit.register(sigterm_handler)
    else:
        signal.signal(signal.SIGTERM, sigterm_handler)
    if sys.platform == 'win32':
        os.kill(0, signal.CTRL_C_EVENT)
    else:
        os.killpg(0, signal.SIGTERM)