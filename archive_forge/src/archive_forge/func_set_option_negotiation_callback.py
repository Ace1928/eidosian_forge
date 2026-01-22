import sys
import socket
import selectors
from time import monotonic as _time
import warnings
def set_option_negotiation_callback(self, callback):
    """Provide a callback function called after each receipt of a telnet option."""
    self.option_callback = callback