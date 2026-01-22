import sys
from types import ModuleType
from typing import Iterable, List, Tuple
from twisted.python.filepath import FilePath
def replaceSysPath(self, sysPath: List[str]) -> None:
    """
        Replace sys.path, for the duration of the test, with the given value.
        """
    originalSysPath = sys.path[:]

    def cleanUpSysPath() -> None:
        sys.path[:] = originalSysPath
    self.addCleanup(cleanUpSysPath)
    sys.path[:] = sysPath