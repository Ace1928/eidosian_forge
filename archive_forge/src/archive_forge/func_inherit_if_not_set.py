import re
import zlib
import base64
from types import MappingProxyType
from numba.core import utils
def inherit_if_not_set(self, name, default=_NotSet):
    """Inherit flag from ``ConfigStack``.

        Parameters
        ----------
        name : str
            Option name.
        default : optional
            When given, it overrides the default value.
            It is only used when the flag is not defined locally and there is
            no entry in the ``ConfigStack``.
        """
    self._guard_option(name)
    if not self.is_set(name):
        cstk = ConfigStack()
        if cstk:
            top = cstk.top()
            setattr(self, name, getattr(top, name))
        elif default is not _NotSet:
            setattr(self, name, default)