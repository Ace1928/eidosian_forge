import logging
import threading
import urllib
import warnings
from contextlib import contextmanager
from dataclasses import fields
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
import requests
from ray.dashboard.modules.dashboard_sdk import SubmissionClient
from ray.dashboard.utils import (
from ray.util.annotations import DeveloperAPI
from ray.util.state.common import (
from ray.util.state.exception import RayStateApiException, ServerUnavailable
@DeveloperAPI
def list_logs(address: Optional[str]=None, node_id: Optional[str]=None, node_ip: Optional[str]=None, glob_filter: Optional[str]=None, timeout: int=DEFAULT_RPC_TIMEOUT) -> Dict[str, List[str]]:
    """Listing log files available.

    Args:
        address: Ray bootstrap address, could be `auto`, `localhost:6379`.
            If not specified, it will be retrieved from the initialized ray cluster.
        node_id: Id of the node containing the logs.
        node_ip: Ip of the node containing the logs.
        glob_filter: Name of the file (relative to the ray log directory) to be
            retrieved. E.g. `glob_filter="*worker*"` for all worker logs.
        actor_id: Id of the actor if getting logs from an actor.
        timeout: Max timeout for requests made when getting the logs.
        _interval: The interval in secs to print new logs when `follow=True`.

    Return:
        A dictionary where the keys are log groups (e.g. gcs, raylet, worker), and
        values are list of log filenames.

    Raises:
        Exceptions: :class:`RayStateApiException <ray.util.state.exception.RayStateApiException>` if the CLI
            failed to query the data, or ConnectionError if failed to resolve the
            ray address.
    """
    assert node_ip is not None or node_id is not None, 'At least one of node ip and node id is required'
    api_server_url = ray_address_to_api_server_url(address)
    if not glob_filter:
        glob_filter = '*'
    options_dict = {}
    if node_ip:
        options_dict['node_ip'] = node_ip
    if node_id:
        options_dict['node_id'] = node_id
    if glob_filter:
        options_dict['glob'] = glob_filter
    options_dict['timeout'] = timeout
    r = requests.get(f'{api_server_url}/api/v0/logs?{urllib.parse.urlencode(options_dict)}')
    r.raise_for_status()
    response = r.json()
    if response['result'] is False:
        raise RayStateApiException(f'API server internal error. See dashboard.log file for more details. Error: {response['msg']}')
    return response['data']['result']