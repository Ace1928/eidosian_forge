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
def handle_join_last_call(self, expected_version, deadline):
    """
        After we reach min number of workers, one particular worker takes on the
        responsibility of waiting an additional timeout before closing the join window.
        If the worker responsible for this fails, the rendezvous will be destroyed due
        to expiring TTL, and the other participants will re-rendezvous.

        Here we expect to see state <joinable, expected_version>
        Exit gracefully if either:

        1. state becomes <frozen, expected_version>
        2. timeout happens (reaching deadline), in which case
           we try the transition to <frozen, expected_version>

        Exit with exception otherwise.
        """
    active_version, state = self.get_rdzv_state()
    while True:
        if state['status'] == 'frozen' and state['version'] == expected_version:
            return
        if state['status'] != 'joinable' or state['version'] != expected_version:
            raise EtcdRendezvousRetryableFailure('Rendezvous state transition no longer possible. Must re-enter.')
        if time.time() >= deadline:
            state['status'] = 'frozen'
            state['keep_alives'] = []
            try:
                active_version = self.client.test_and_set(key=self.get_path('/rdzv/active_version'), value=json.dumps(state), prev_value=active_version.value, ttl=CONST_ETCD_FROZEN_TTL)
                return
            except etcd.EtcdCompareFailed:
                log.info('Join last-call transition CAS unsuccessful. Will retry')
                cas_delay()
                active_version, state = self.get_rdzv_state()
                continue
        try:
            active_version = self.client.test_and_set(key=self.get_path('/rdzv/active_version'), value=active_version.value, prev_value=active_version.value, ttl=CONST_ETCD_JOINABLE_EPHEMERAL_TTL)
            timeout = min(CONST_ETCD_JOINABLE_EPHEMERAL_TTL / 2, deadline - time.time() + 1.0)
            active_version, state = self.try_wait_for_state_change(etcd_index=active_version.etcd_index + 1, timeout=timeout)
        except etcd.EtcdCompareFailed:
            log.info('Join last-call TTL refresh CAS unsuccessful, will retry')
            cas_delay()
            active_version, state = self.get_rdzv_state()