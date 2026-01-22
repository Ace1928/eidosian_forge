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
def create_rdzv_handler(params: RendezvousParameters) -> RendezvousHandler:
    """
    Usage:

    ::

    rdzv_params = RendezvousParameters(
                        backend="etcd",
                        endpoint="192.168.0.42:2379",
                        run_id="123",
                        min_nodes=4,
                        max_nodes=8,
                        timeout=300,
                        last_call_timeout=30,
                        etcd_prefix="custom_prefix",
                        protocol="https",
                        cacert="/etc/kubernetes/certs/ca.crt",
                        cert="/etc/kubernetes/certs/client.crt",
                        key="/etc/kubernetes/certs/client.key")
    # -- or --
    rdzv_params = RendezvousParameters(
                        backend="etcd",
                        endpoint="192.168.0.42:2379",
                        run_id="123",
                        min_nodes=4,
                        max_nodes=8)

    etcd_rdzv_handler = create_etcd_rendezvous_handler(rdzv_params)


    Where:
        run_id - unique id for this training job instance,
        min_nodes - min number of workers expected to join the rendezvous,
        max_nodes - max number of workers allowed to join the rendezvous,
                        defaults to min_workers is not specified.
        timeout - total timeout within which next_rendezvous is expected to
                      succeed; a RendezvousTimeoutError is raised otherwise;
                      Defaults is 600 (10 minutes).
        last_call_timeout - additional wait amount ("last call") after
                            min number of workers has been reached.
                            Defaults to 30 seconds.
        etcd_prefix - path prefix (from etcd root), inside which all
                      etcd nodes will be created.
                      Default is "/torchelastic/p2p".
        protocol - http (default) or https to access etcd.
        cacert - CA cert to access etcd, only makes sense with https.
        cert - client cert to access etcd, only makes sense with https.
        key - client key to access etcd, only makes sense with https.
    """
    client = _create_etcd_client(params)
    etcd_prefix = params.get('etcd_prefix', '/torchelastic/p2p')
    rdzv = EtcdRendezvous(client=client, prefix=etcd_prefix, run_id=params.run_id, num_min_workers=params.min_nodes, num_max_workers=params.max_nodes, timeout=params.get_as_int('timeout', _DEFAULT_TIMEOUT), last_call_timeout=params.get_as_int('last_call_timeout', _DEFAULT_LAST_CALL_TIMEOUT))
    return EtcdRendezvousHandler(rdzv_impl=rdzv)