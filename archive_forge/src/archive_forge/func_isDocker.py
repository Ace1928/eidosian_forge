import os
import sys
import warnings
from time import time as seconds
from typing import Optional
def isDocker(self, _initCGroupLocation: str='/proc/1/cgroup') -> bool:
    """
        Check if the current platform is Linux in a Docker container.

        @return: C{True} if the current platform has been detected as Linux
            inside a Docker container.
        """
    if not self.isLinux():
        return False
    from twisted.python.filepath import FilePath
    initCGroups = FilePath(_initCGroupLocation)
    if initCGroups.exists():
        controlGroups = [x.split(b':') for x in initCGroups.getContent().split(b'\n')]
        for group in controlGroups:
            if len(group) == 3 and group[2].startswith(b'/docker/'):
                return True
    return False