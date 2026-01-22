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
def summarize_actors(address: Optional[str]=None, timeout: int=DEFAULT_RPC_TIMEOUT, raise_on_missing_output: bool=True, _explain: bool=False) -> Dict:
    """Summarize the actors in cluster.

    Args:
        address: Ray bootstrap address, could be `auto`, `localhost:6379`.
            If None, it will be resolved automatically from an initialized ray.
        timeout: Max timeout for requests made when getting the states.
        raise_on_missing_output: When True, exceptions will be raised if
            there is missing data due to truncation/data source unavailable.
        _explain: Print the API information such as API latency or
            failed query information.

    Return:
        Dictionarified
        :class:`~ray.util.state.common.ActorSummaries`

    Raises:
        Exceptions: :class:`RayStateApiException <ray.util.state.exception.RayStateApiException>` if the CLI
            failed to query the data.
    """
    return StateApiClient(address=address).summary(SummaryResource.ACTORS, options=SummaryApiOptions(timeout=timeout), raise_on_missing_output=raise_on_missing_output, _explain=_explain)