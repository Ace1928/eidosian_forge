import socket
import selectors
import os
import sys
import threading
from io import BufferedIOBase
from time import monotonic as time
def handle_timeout(self):
    """Wait for zombies after self.timeout seconds of inactivity.

            May be extended, do not override.
            """
    self.collect_children()