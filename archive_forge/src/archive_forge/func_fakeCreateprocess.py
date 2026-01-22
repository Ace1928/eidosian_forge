import errno
import gc
import gzip
import operator
import os
import signal
import stat
import sys
from unittest import SkipTest, skipIf
from io import BytesIO
from zope.interface.verify import verifyObject
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.python import procutils, runtime
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath
from twisted.python.log import msg
from twisted.trial import unittest
def fakeCreateprocess(appName, commandLine, processAttributes, threadAttributes, bInheritHandles, creationFlags, newEnvironment, currentDirectory, startupinfo):
    """
            See the Windows API documentation for I{CreateProcess} for further details.

            @param appName: The name of the module to be executed
            @param commandLine: The command line to be executed.
            @param processAttributes: Pointer to SECURITY_ATTRIBUTES structure or None.
            @param threadAttributes: Pointer to SECURITY_ATTRIBUTES structure or  None
            @param bInheritHandles: boolean to determine if inheritable handles from this
                                    process are inherited in the new process
            @param creationFlags: flags that control priority flags and creation of process.
            @param newEnvironment: pointer to new environment block for new process, or None.
            @param currentDirectory: full path to current directory of new process.
            @param startupinfo: Pointer to STARTUPINFO or STARTUPINFOEX structure
            @return: True on success, False on failure
            @rtype: L{bool}
            """
    flags.append(creationFlags)
    return realCreateProcess(appName, commandLine, processAttributes, threadAttributes, bInheritHandles, creationFlags, newEnvironment, currentDirectory, startupinfo)