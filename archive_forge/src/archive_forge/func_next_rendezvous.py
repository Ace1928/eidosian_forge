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
def next_rendezvous(self):
    rdzv_version, rank, world_size = self._rdzv_impl.rendezvous_barrier()
    log.info('Creating EtcdStore as the c10d::Store implementation')
    store = self._rdzv_impl.setup_kv_store(rdzv_version)
    return (store, rank, world_size)