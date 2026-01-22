from .. import errors
from ..decorators import only_raises
def disable_lock_write(self):
    """Make a lock_write call fail"""
    self.__dict__['_allow_write'] = False