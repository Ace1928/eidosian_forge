import asyncio
from jupyter_client.client import KernelClient
from jupyter_client.clientabc import KernelClientABC
from jupyter_core.utils import run_sync
from traitlets import Instance, Type, default
from .channels import InProcessChannel, InProcessHBChannel
def get_shell_msg(self, block=True, timeout=None):
    """Get a shell message."""
    return self.shell_channel.get_msg(block, timeout)