import sys
from IPython import get_ipython
import comm
def get_comm_manager():
    if requires_ipykernel_shim():
        ip = get_ipython()
        if ip is not None and getattr(ip, 'kernel', None) is not None:
            return get_ipython().kernel.comm_manager
    else:
        return comm.get_comm_manager()