from __future__ import annotations
import sys
from queue import Empty, Queue
from kombu.exceptions import reraise
from kombu.log import get_logger
from kombu.utils.objects import cached_property
from . import virtual
Kombu Broker used by the Pyro transport.

        You have to run this as a separate (Pyro) service.
        