from functools import partial
from glob import glob
from distutils.util import convert_path
import distutils.command.build_py as orig
import os
import fnmatch
import textwrap
import io
import distutils.errors
import itertools
import stat
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple
from ..extern.more_itertools import unique_everseen
from ..warnings import SetuptoolsDeprecationWarning
def get_data_files_without_manifest(self):
    """
        Generate list of ``(package,src_dir,build_dir,filenames)`` tuples,
        but without triggering any attempt to analyze or build the manifest.
        """
    self.__dict__.setdefault('manifest_files', {})
    return list(map(self._get_pkg_data_files, self.packages or ()))