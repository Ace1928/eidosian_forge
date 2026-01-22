import sys
import signal
import os
import time
from _thread import interrupt_main  # Py 3
import threading
import pytest
from IPython.utils.process import (find_cmd, FindCmdError, arg_split,
from IPython.utils.capture import capture_output
from IPython.testing import decorators as dec
from IPython.testing import tools as tt
def assert_interrupts(self, command):
    """
        Interrupt a subprocess after a second.
        """
    if threading.main_thread() != threading.current_thread():
        raise pytest.skip("Can't run this test if not in main thread.")
    signal.signal(signal.SIGINT, signal.default_int_handler)

    def interrupt():
        time.sleep(0.5)
        interrupt_main()
    threading.Thread(target=interrupt).start()
    start = time.time()
    try:
        result = command()
    except KeyboardInterrupt:
        pass
    end = time.time()
    self.assertTrue(end - start < 2, "Process didn't die quickly: %s" % (end - start))
    return result