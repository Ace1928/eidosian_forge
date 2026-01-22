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
def list_runtime_envs(address: Optional[str]=None, filters: Optional[List[Tuple[str, PredicateType, SupportedFilterType]]]=None, limit: int=DEFAULT_LIMIT, timeout: int=DEFAULT_RPC_TIMEOUT, detail: bool=False, raise_on_missing_output: bool=True, _explain: bool=False) -> List[RuntimeEnvState]:
    """List runtime environments in the cluster.

    Args:
        address: Ray bootstrap address, could be `auto`, `localhost:6379`.
            If None, it will be resolved automatically from an initialized ray.
        filters: List of tuples of filter key, predicate (=, or !=), and
            the filter value. E.g., `("node_id", "=", "abcdef")`
            String filter values are case-insensitive.
        limit: Max number of entries returned by the state backend.
        timeout: Max timeout value for the state APIs requests made.
        detail: When True, more details info (specified in `RuntimeEnvState`)
            will be queried and returned. See
            :class:`RuntimeEnvState <ray.util.state.common.RuntimeEnvState>`.
        raise_on_missing_output: When True, exceptions will be raised if
            there is missing data due to truncation/data source unavailable.
        _explain: Print the API information such as API latency or
            failed query information.

    Returns:
        List of
        :class:`RuntimeEnvState <ray.util.state.common.RuntimeEnvState>`.

    Raises:
        Exceptions: :class:`RayStateApiException <ray.util.state.exception.RayStateApiException>`
            if the CLI failed to query the data.
    """
    return StateApiClient(address=address).list(StateResource.RUNTIME_ENVS, options=ListApiOptions(limit=limit, timeout=timeout, filters=filters, detail=detail), raise_on_missing_output=raise_on_missing_output, _explain=_explain)