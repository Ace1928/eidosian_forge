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
def target_cpu(self):
    """
        Return Target CPU architecture.

        Return
        ------
        str
            Target CPU
        """
    return self.arch[self.arch.find('_') + 1:]