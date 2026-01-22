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
def setup_lease_renewal(self, full_path, ttl):

    def lease_worker(client, path, ttl, stop_event):
        while True:
            try:
                client.refresh(path, ttl=ttl)
            except etcd.EtcdKeyNotFound:
                break
            except ConnectionRefusedError:
                break
            if stop_event.wait(timeout=ttl / 2):
                break
    lease_stop_event = threading.Event()
    lease_thread = threading.Thread(target=lease_worker, args=(self.client, full_path, ttl, lease_stop_event))
    lease_thread.daemon = True
    lease_thread.start()
    return lease_stop_event