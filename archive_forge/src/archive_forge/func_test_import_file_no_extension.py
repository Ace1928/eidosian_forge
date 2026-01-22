import ctypes
import logging
import os
import platform
import shutil
import stat
import sys
import tempfile
import subprocess
from io import StringIO
import pyomo.common.unittest as unittest
import pyomo.common.envvar as envvar
from pyomo.common.log import LoggingIntercept
from pyomo.common.fileutils import (
from pyomo.common.download import FileDownloader
def test_import_file_no_extension(self):
    with self.assertRaises(FileNotFoundError) as context:
        import_file(os.path.join(_this_file_dir, 'import_ex'))
    self.assertTrue('File does not exist' in str(context.exception))