import os
from subprocess import STDOUT, CalledProcessError, check_output
from typing import Dict
from zope.interface import Interface, implementer
from twisted.python.compat import execfile
def getRepositoryCommand(directory):
    """
    Detect the VCS used in the specified directory and return a L{GitCommand}
    if the directory is a Git repository. If the directory is not git, it
    raises a L{NotWorkingDirectory} exception.

    @type directory: L{FilePath}
    @param directory: The directory to detect the VCS used from.

    @rtype: L{GitCommand}

    @raise NotWorkingDirectory: if no supported VCS can be found from the
        specified directory.
    """
    try:
        GitCommand.ensureIsWorkingDirectory(directory)
        return GitCommand
    except (NotWorkingDirectory, OSError):
        pass
    raise NotWorkingDirectory(f'No supported VCS can be found in {directory.path}')