import random
import threading
import time
from .messages import Message
from .parser import Parser
def multi_iter_pending(ports, yield_ports=False):
    """Iterate through all pending messages in ports.

    This is the same as calling multi_receive(ports, block=False).
    The function is kept around for backwards compatability.
    """
    return multi_receive(ports, yield_ports=yield_ports, block=False)