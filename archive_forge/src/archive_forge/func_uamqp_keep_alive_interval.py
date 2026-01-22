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
def uamqp_keep_alive_interval(self) -> int:
    return self.transport_options.get('uamqp_keep_alive_interval', self.default_uamqp_keep_alive_interval)