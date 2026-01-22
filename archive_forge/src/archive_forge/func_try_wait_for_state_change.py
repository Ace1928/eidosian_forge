import json
import logging
import sys
import threading
import time
from typing import Optional
import etcd  # type: ignore[import]
from torch.distributed.elastic.rendezvous import (
from .utils import parse_rendezvous_endpoint
from .etcd_store import EtcdStore, cas_delay
def try_wait_for_state_change(self, etcd_index, timeout=None):
    overall_timeout = max(self._rendezvous_deadline - time.time(), 0.0) + 1.0
    timeout = overall_timeout if timeout is None else min(timeout, overall_timeout)
    try:
        self.client.watch(self.get_path('/rdzv/active_version'), index=etcd_index, timeout=timeout)
    except (etcd.EtcdEventIndexCleared, etcd.EtcdWatchTimedOut):
        pass
    if time.time() > self._rendezvous_deadline:
        raise RendezvousTimeoutError()
    return self.get_rdzv_state()