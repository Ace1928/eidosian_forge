import os
from typing import Any, Callable, Dict, List, Tuple, Type
from .. import conflicts, errors, option, osutils, tests, transform
from ..bzr import conflicts as bzr_conflicts
from ..workingtree import WorkingTree
from . import scenarios, script
def check_file_in_dir_has_more_content(self):
    self.assertFileEqual(b'trunk content\nmore content\n', 'branch/dir/file')