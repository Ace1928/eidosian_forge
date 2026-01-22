import asyncio
from jupyter_client.client import KernelClient
from jupyter_client.clientabc import KernelClientABC
from jupyter_core.utils import run_sync
from traitlets import Instance, Type, default
from .channels import InProcessChannel, InProcessHBChannel
def start_channels(self, *args, **kwargs):
    """Start the channels on the client."""
    super().start_channels()
    if self.kernel:
        self.kernel.frontends.append(self)