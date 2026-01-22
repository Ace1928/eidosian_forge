import os
from typing import Any, Callable, Dict, List, Tuple, Type
from .. import conflicts, errors, option, osutils, tests, transform
from ..bzr import conflicts as bzr_conflicts
from ..workingtree import WorkingTree
from . import scenarios, script
def check_resolved(self, wt, action):
    path = self._get_resolve_path_arg(wt, action)
    conflicts.resolve(wt, [path], action=action)
    self.assertLength(0, wt.conflicts())
    self.assertLength(0, list(wt.unknowns()))