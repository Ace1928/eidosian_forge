import sys
from queue import Empty, Queue
from traitlets import Type
from .channels import InProcessChannel
from .client import InProcessKernelClient
def msg_ready(self):
    """Is there a message that has been received?"""
    return not self._in_queue.empty()