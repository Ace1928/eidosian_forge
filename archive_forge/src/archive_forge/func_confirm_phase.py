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
def confirm_phase(self, expected_version, this_rank):
    """
        Once the rendezvous state transitions from 'joinable' to 'frozen',
        we have every participant confirm their membership and setup per-member
        keep-alive TTL keys, and then wait for all other participants to confirm,
        which would then successfully conclude this rendezvous.
        """
    log.info('All peers arrived. Confirming membership.')
    self.confirm_membership(expected_version, this_rank)
    log.info('Waiting for confirmations from all peers.')
    active_version = self.wait_for_final(expected_version)
    state = json.loads(active_version.value)
    log.info('Rendezvous version %s is complete. Final state: %s', state['version'], state)
    return (state['version'], this_rank, len(state['participants']))