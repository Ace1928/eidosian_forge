import collections
import platform
import sys
def include_system(self):
    """Append the information about the Operating System."""
    self._pieces.append(_platform_tuple())
    return self