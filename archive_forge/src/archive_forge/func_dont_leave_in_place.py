from typing import Optional, Type, Union
from . import counted_lock, errors, lock, transactions, urlutils
from .decorators import only_raises
from .transport import Transport
def dont_leave_in_place(self):
    raise NotImplementedError(self.dont_leave_in_place)