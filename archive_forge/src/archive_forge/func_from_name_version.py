from contextlib import contextmanager
import glob
from importlib import import_module
import io
import itertools
import os.path as osp
import re
import sys
import warnings
import zipfile
import configparser
@classmethod
def from_name_version(cls, name):
    """Parse a distribution from a "name-version" string

        :param str name: The name-version string (entrypoints-0.3)
        Returns an :class:`Distribution` object
        """
    version = None
    if '-' in name:
        name, version = name.split('-', 1)
    return cls(name, version)