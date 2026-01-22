from jupyter_client.manager import KernelManager
from jupyter_client.managerabc import KernelManagerABC
from jupyter_client.session import Session
from traitlets import DottedObjectName, Instance, default
from .constants import INPROCESS_KEY
def start_kernel(self, **kwds):
    """Start the kernel."""
    from ipykernel.inprocess.ipkernel import InProcessKernel
    self.kernel = InProcessKernel(parent=self, session=self.session)