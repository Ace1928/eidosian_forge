import os
from collections import defaultdict
from typing import Callable, Dict, Iterable, Sequence, Set, List
from .. import importcompletion
def on_any_event(self, event: FileSystemEvent) -> None:
    dirpath = os.path.dirname(event.src_path)
    if any((event.src_path == f'{path}.py' for path in self.dirs[dirpath])):
        self.on_change((event.src_path,))