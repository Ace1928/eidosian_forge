import threading
from typing import MutableMapping, NamedTuple
import wandb
def update_failed_file(self, save_name: str) -> None:
    with self._lock:
        self._stats[save_name] = self._stats[save_name]._replace(uploaded=0, failed=True)