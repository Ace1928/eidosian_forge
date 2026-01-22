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
def load_extra_data(self, rdzv_version, key, timeout=None):
    node = self.get_path(f'/rdzv/v_{rdzv_version}/extra_data')
    node_dir = self.get_path(f'/rdzv/v_{rdzv_version}')
    while True:
        root = self.client.get(node_dir)
        extra_data = [n for n in root.children if n.key == node]
        assert len(extra_data) <= 1
        if len(extra_data) == 1:
            extra_data_dict = json.loads(extra_data[0].value)
            if key in extra_data_dict:
                return extra_data_dict[key]
        try:
            self.client.watch(node, index=root.etcd_index + 1)
        except (etcd.EtcdEventIndexCleared, etcd.EtcdWatchTimedOut):
            pass