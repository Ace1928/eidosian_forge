from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import emulation
from . import io
from . import page
from . import runtime
from . import security
def emulate_network_conditions(offline: bool, latency: float, download_throughput: float, upload_throughput: float, connection_type: typing.Optional[ConnectionType]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Activates emulation of network conditions.

    :param offline: True to emulate internet disconnection.
    :param latency: Minimum latency from request sent to response headers received (ms).
    :param download_throughput: Maximal aggregated download throughput (bytes/sec). -1 disables download throttling.
    :param upload_throughput: Maximal aggregated upload throughput (bytes/sec).  -1 disables upload throttling.
    :param connection_type: *(Optional)* Connection type if known.
    """
    params: T_JSON_DICT = dict()
    params['offline'] = offline
    params['latency'] = latency
    params['downloadThroughput'] = download_throughput
    params['uploadThroughput'] = upload_throughput
    if connection_type is not None:
        params['connectionType'] = connection_type.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Network.emulateNetworkConditions', 'params': params}
    json = (yield cmd_dict)