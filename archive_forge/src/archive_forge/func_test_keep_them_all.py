import os
from typing import Any, Callable, Dict, List, Tuple, Type
from .. import conflicts, errors, option, osutils, tests, transform
from ..bzr import conflicts as bzr_conflicts
from ..workingtree import WorkingTree
from . import scenarios, script
def test_keep_them_all(self):
    self.run_script("\n$ brz resolve dir\n2>2 conflicts resolved, 0 remaining\n$ brz commit -q --strict -m 'No more conflicts nor unknown files'\n")