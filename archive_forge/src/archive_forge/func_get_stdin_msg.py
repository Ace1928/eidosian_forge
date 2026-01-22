import asyncio
from jupyter_client.client import KernelClient
from jupyter_client.clientabc import KernelClientABC
from jupyter_core.utils import run_sync
from traitlets import Instance, Type, default
from .channels import InProcessChannel, InProcessHBChannel
def get_stdin_msg(self, block=True, timeout=None):
    """Get a stdin message."""
    return self.stdin_channel.get_msg(block, timeout)