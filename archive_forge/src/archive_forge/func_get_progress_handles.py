import secrets
import string
import threading
import time
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple
from wandb.errors import Error
from wandb.proto import wandb_internal_pb2 as pb
def get_progress_handles(self) -> List[MailboxProgress]:
    return [ph for ph in self._progress_handles if not ph._handle._is_failed]