import asyncio
from contextlib import contextmanager
from typing import Iterator
import httpx
from qcs_api_client.client import QCSClientConfiguration, build_sync_client
@contextmanager
def qcs_client(*, client_configuration: QCSClientConfiguration, request_timeout: float=10.0) -> Iterator[httpx.Client]:
    """
    Build a QCS client.

    :param client_configuration: Client configuration.
    :param request_timeout: Time limit for requests, in seconds.
    """
    _ensure_event_loop()
    with build_sync_client(configuration=client_configuration, client_kwargs={'timeout': request_timeout}) as client:
        yield client