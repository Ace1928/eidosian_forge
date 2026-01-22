from __future__ import annotations
import string
from queue import Empty
from typing import Any, Dict, Set
import azure.core.exceptions
import azure.servicebus.exceptions
import isodate
from azure.servicebus import (ServiceBusClient, ServiceBusMessage,
from azure.servicebus.management import ServiceBusAdministrationClient
from kombu.utils.encoding import bytes_to_str, safe_str
from kombu.utils.json import dumps, loads
from kombu.utils.objects import cached_property
from . import virtual
@cached_property
def queue_service(self) -> ServiceBusClient:
    if self._connection_string:
        return ServiceBusClient.from_connection_string(self._connection_string, retry_total=self.retry_total, retry_backoff_factor=self.retry_backoff_factor, retry_backoff_max=self.retry_backoff_max)
    return ServiceBusClient(self._namespace, self._credential, retry_total=self.retry_total, retry_backoff_factor=self.retry_backoff_factor, retry_backoff_max=self.retry_backoff_max)