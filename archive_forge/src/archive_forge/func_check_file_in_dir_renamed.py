import os
from typing import Any, Callable, Dict, List, Tuple, Type
from .. import conflicts, errors, option, osutils, tests, transform
from ..bzr import conflicts as bzr_conflicts
from ..workingtree import WorkingTree
from . import scenarios, script
def check_file_in_dir_renamed(self):
    self.assertPathDoesNotExist('branch/dir/file')
    self.assertPathExists('branch/dir/new-file')