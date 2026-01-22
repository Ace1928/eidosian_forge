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
def get_rdzv_state(self):
    active_version = self.client.get(key=self.get_path('/rdzv/active_version'))
    return (active_version, json.loads(active_version.value))