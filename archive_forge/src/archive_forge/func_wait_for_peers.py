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
def wait_for_peers(self, expected_version):
    """Helper method for the join phase."""
    active_version, state = self.get_rdzv_state()
    while True:
        if state['status'] == 'frozen' and state['version'] == expected_version:
            return active_version
        elif state['status'] == 'joinable' and state['version'] == expected_version:
            active_version, state = self.try_wait_for_state_change(etcd_index=active_version.etcd_index + 1)
        else:
            raise EtcdRendezvousRetryableFailure('Rendezvous state transition no longer possible. Must re-enter.')