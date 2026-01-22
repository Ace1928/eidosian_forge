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
def handle_existing_rendezvous(self, expected_version):
    """
        Handle the case when there's an existing (state 'final) rendezvous already
        in place, and we have to announce ourselves waiting, and wait until
        the next rendezvous opportunity.
        """
    active_state = self.announce_self_waiting(expected_version)
    log.info('Added self to waiting list. Rendezvous full state: %s', active_state.value)
    self.wait_for_rendezvous_to_free(expected_version)
    log.info('Previously existing rendezvous state changed. Will re-try joining.')