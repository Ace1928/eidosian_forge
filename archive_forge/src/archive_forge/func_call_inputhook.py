from __future__ import unicode_literals
import os
import threading
from prompt_toolkit.utils import is_windows
from .select import select_fds
def call_inputhook(self, input_is_ready_func):
    """
        Call the inputhook. (Called by a prompt-toolkit eventloop.)
        """
    self._input_is_ready = input_is_ready_func

    def thread():
        input_is_ready_func(wait=True)
        os.write(self._w, b'x')
    threading.Thread(target=thread).start()
    self.inputhook(self)
    try:
        if not is_windows():
            select_fds([self._r], timeout=None)
        os.read(self._r, 1024)
    except OSError:
        pass
    self._input_is_ready = None