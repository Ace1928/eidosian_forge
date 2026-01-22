import atexit
import ssl
from pyVim.connect import Disconnect, SmartStubAdapter, VimSessionOrientedStub
from pyVmomi import vim
from ray.autoscaler._private.vsphere.utils import Constants, singleton_client
def get_client(self):
    if self.session_type == Constants.SessionType.UNVERIFIED:
        context_obj = ssl._create_unverified_context()
    else:
        pass
    credentials = VimSessionOrientedStub.makeUserLoginMethod(self.user, self.password)
    smart_stub = SmartStubAdapter(host=self.server, port=self.port, sslContext=context_obj, connectionPoolTimeout=self.timeout)
    session_stub = VimSessionOrientedStub(smart_stub, credentials)
    return vim.ServiceInstance('ServiceInstance', session_stub)