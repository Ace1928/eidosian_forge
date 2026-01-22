from __future__ import print_function, unicode_literals
import typing
import six
from ._typing import Text
def to_platform_bin(self):
    """Get a *binary* mode string for the current platform.

        This removes the 't' and adds a 'b' if needed.

        """
    _mode = self.to_platform().replace('t', '')
    return _mode if 'b' in _mode else _mode + 'b'