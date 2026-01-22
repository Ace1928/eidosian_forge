from __future__ import annotations
import threading
from typing import Callable
from oslo_concurrency import processutils as putils
from oslo_context import context as context_utils
from oslo_utils import encodeutils
from os_brick.privileged import rootwrap as priv_rootwrap
def set_root_helper(self, helper: str) -> None:
    self._root_helper = helper