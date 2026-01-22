import os
import stat
import sys
import unittest.mock
from test.support import unix_shell, requires_subprocess
from test.support import os_helper
from distutils.spawn import find_executable
from distutils.spawn import spawn
from distutils.errors import DistutilsExecError
from distutils.tests import support
Tests for distutils.spawn.