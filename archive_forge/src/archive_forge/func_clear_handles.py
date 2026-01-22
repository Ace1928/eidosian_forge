import secrets
import string
import threading
import time
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple
from wandb.errors import Error
from wandb.proto import wandb_internal_pb2 as pb
def clear_handles(self) -> None:
    for handle in self._handles:
        handle._slot._clear_wait_all()
    self._handles = []