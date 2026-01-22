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
def queue_mgmt_service(self) -> ServiceBusAdministrationClient:
    if self._connection_string:
        return ServiceBusAdministrationClient.from_connection_string(self._connection_string)
    return ServiceBusAdministrationClient(self._namespace, self._credential)