from jupyter_client.manager import KernelManager
from jupyter_client.managerabc import KernelManagerABC
from jupyter_client.session import Session
from traitlets import DottedObjectName, Instance, default
from .constants import INPROCESS_KEY
def interrupt_kernel(self):
    """Interrupt the kernel."""
    msg = 'Cannot interrupt in-process kernel.'
    raise NotImplementedError(msg)