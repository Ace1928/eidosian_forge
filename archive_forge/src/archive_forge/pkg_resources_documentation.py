from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import fnmatch
import glob
import importlib.util
import os
import pkgutil
import sys
import types
from googlecloudsdk.core.util import files
Get files from a given directory that match a pattern.

  Args:
    path_dir: str, filesystem path to directory
    filter_pattern: str, pattern to filter files to retrieve.

  Returns:
    List of filtered files from a directory.

  Raises:
    IOError: if resource is not found under given path.
  