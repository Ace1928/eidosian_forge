import sys
from queue import Empty, Queue
from traitlets import Type
from .channels import InProcessChannel
from .client import InProcessKernelClient
def call_handlers(self, msg):
    """Overridden for the in-process channel.

        This methods simply calls raw_input directly.
        """
    msg_type = msg['header']['msg_type']
    if msg_type == 'input_request':
        _raw_input = self.client.kernel._sys_raw_input
        prompt = msg['content']['prompt']
        print(prompt, end='', file=sys.__stdout__)
        sys.__stdout__.flush()
        self.client.input(_raw_input())