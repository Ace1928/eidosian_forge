from __future__ import annotations
import random
from contextlib import asynccontextmanager
from itertools import count
from typing import TYPE_CHECKING, NoReturn
import attrs
import pytest
from trio._tests.pytest_plugin import skip_if_optional_else_raise
import trio
import trio.testing
from trio import DTLSChannel, DTLSEndpoint
from trio.testing._fake_net import FakeNet, UDPPacket
from .._core._tests.tutil import binds_ipv6, gc_collect_harder, slow
def route_packet_wrapper(packet: UDPPacket) -> None:
    try:
        nursery.start_soon(route_packet, packet)
    except RuntimeError:
        pass