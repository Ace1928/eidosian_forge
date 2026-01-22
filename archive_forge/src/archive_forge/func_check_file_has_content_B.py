import os
from typing import Any, Callable, Dict, List, Tuple, Type
from .. import conflicts, errors, option, osutils, tests, transform
from ..bzr import conflicts as bzr_conflicts
from ..workingtree import WorkingTree
from . import scenarios, script
def check_file_has_content_B(self, path='file'):
    self.assertFileEqual(b'trunk content\nfeature B\n', osutils.pathjoin('branch', path))