from __future__ import unicode_literals
from prompt_toolkit.buffer import EditReadOnlyBuffer
from prompt_toolkit.filters.cli import ViNavigationMode
from prompt_toolkit.keys import Keys, Key
from prompt_toolkit.utils import Event
from .registry import BaseRegistry
from collections import deque
from six.moves import range
import weakref
import six
def process_keys(self):
    """
        Process all the keys in the `input_queue`.
        (To be called after `feed`.)

        Note: because of the `feed`/`process_keys` separation, it is
              possible to call `feed` from inside a key binding.
              This function keeps looping until the queue is empty.
        """
    while self.input_queue:
        key_press = self.input_queue.popleft()
        if key_press.key != Keys.CPRResponse:
            self.beforeKeyPress.fire()
        self._process_coroutine.send(key_press)
        if key_press.key != Keys.CPRResponse:
            self.afterKeyPress.fire()
    cli = self._cli_ref()
    if cli:
        cli.invalidate()