import json
from os import listdir, pathsep
from os.path import join, isfile, isdir, dirname
from subprocess import CalledProcessError
import contextlib
import platform
import itertools
import subprocess
import distutils.errors
from setuptools.extern.more_itertools import unique_everseen
@property
def vs_ver(self):
    """
        Microsoft Visual Studio.

        Return
        ------
        float
            version
        """
    return self.si.vs_ver