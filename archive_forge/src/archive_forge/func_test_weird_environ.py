import collections
import errno
import getpass
import itertools
import os
import signal
import socket
import stat
import subprocess
import sys
import textwrap
import time
import types
import unittest
import psutil
from psutil import AIX
from psutil import BSD
from psutil import LINUX
from psutil import MACOS
from psutil import NETBSD
from psutil import OPENBSD
from psutil import OSX
from psutil import POSIX
from psutil import SUNOS
from psutil import WINDOWS
from psutil._common import open_text
from psutil._compat import PY3
from psutil._compat import FileNotFoundError
from psutil._compat import long
from psutil._compat import super
from psutil.tests import APPVEYOR
from psutil.tests import CI_TESTING
from psutil.tests import GITHUB_ACTIONS
from psutil.tests import GLOBAL_TIMEOUT
from psutil.tests import HAS_CPU_AFFINITY
from psutil.tests import HAS_ENVIRON
from psutil.tests import HAS_IONICE
from psutil.tests import HAS_MEMORY_MAPS
from psutil.tests import HAS_PROC_CPU_NUM
from psutil.tests import HAS_PROC_IO_COUNTERS
from psutil.tests import HAS_RLIMIT
from psutil.tests import HAS_THREADS
from psutil.tests import MACOS_11PLUS
from psutil.tests import PYPY
from psutil.tests import PYTHON_EXE
from psutil.tests import PYTHON_EXE_ENV
from psutil.tests import PsutilTestCase
from psutil.tests import ThreadTask
from psutil.tests import call_until
from psutil.tests import copyload_shared_lib
from psutil.tests import create_c_exe
from psutil.tests import create_py_exe
from psutil.tests import mock
from psutil.tests import process_namespace
from psutil.tests import reap_children
from psutil.tests import retry_on_failure
from psutil.tests import sh
from psutil.tests import skip_on_access_denied
from psutil.tests import skip_on_not_implemented
from psutil.tests import wait_for_pid
@unittest.skipIf(not HAS_ENVIRON, 'not supported')
@unittest.skipIf(not POSIX, 'POSIX only')
@unittest.skipIf(MACOS_11PLUS, "macOS 11+ can't get another process environment, issue #2084")
def test_weird_environ(self):
    code = textwrap.dedent('\n            #include <unistd.h>\n            #include <fcntl.h>\n\n            char * const argv[] = {"cat", 0};\n            char * const envp[] = {"A=1", "X", "C=3", 0};\n\n            int main(void) {\n                // Close stderr on exec so parent can wait for the\n                // execve to finish.\n                if (fcntl(2, F_SETFD, FD_CLOEXEC) != 0)\n                    return 0;\n                return execve("/bin/cat", argv, envp);\n            }\n            ')
    cexe = create_c_exe(self.get_testfn(), c_code=code)
    sproc = self.spawn_testproc([cexe], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    p = psutil.Process(sproc.pid)
    wait_for_pid(p.pid)
    assert p.is_running()
    self.assertEqual(sproc.stderr.read(), b'')
    if MACOS and CI_TESTING:
        try:
            env = p.environ()
        except psutil.AccessDenied:
            return
    else:
        env = p.environ()
    self.assertEqual(env, {'A': '1', 'C': '3'})
    sproc.communicate()
    self.assertEqual(sproc.returncode, 0)