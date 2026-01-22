import sys
from IPython import get_ipython
import comm
def requires_ipykernel_shim():
    if 'ipykernel' in sys.modules:
        import ipykernel
        version = ipykernel.version_info
        return version < (6, 18)
    else:
        return False