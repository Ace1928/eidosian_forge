import unittest
import sys
import os
from test.support import requires_zlib
from distutils.core import Distribution
from distutils.command.bdist_rpm import bdist_rpm
from distutils.tests import support
from distutils.spawn import find_executable
from distutils.core import setup
import foo
Tests for distutils.command.bdist_rpm.