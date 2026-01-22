from torch._C import _set_backcompat_broadcast_warn
from torch._C import _get_backcompat_broadcast_warn
from torch._C import _set_backcompat_keepdim_warn
from torch._C import _get_backcompat_keepdim_warn
def set_enabled(self, value):
    self.setter(value)