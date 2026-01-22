from torch._C import _set_backcompat_broadcast_warn
from torch._C import _get_backcompat_broadcast_warn
from torch._C import _set_backcompat_keepdim_warn
from torch._C import _get_backcompat_keepdim_warn
def get_enabled(self):
    return self.getter()