from __future__ import annotations
import contextlib
import errno
import ipaddress
import os
import socket
import sys
from typing import (
import attrs
import trio
from trio._util import NoPublicConstructor, final
def deliver_packet(self, packet: UDPPacket) -> None:
    binding = UDPBinding(local=packet.destination)
    if binding in self._bound:
        self._bound[binding]._deliver_packet(packet)
    else:
        pass