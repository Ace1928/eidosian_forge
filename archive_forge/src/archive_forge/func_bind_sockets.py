from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Sequence
from tornado import netutil
def bind_sockets(address: str | None, port: int) -> tuple[list[socket], int]:
    """ Bind a socket to a port on an address.

    Args:
        address (str) :
            An address to bind a port on, e.g. ``"localhost"``

        port (int) :
            A port number to bind.

            Pass 0 to have the OS automatically choose a free port.

    This function returns a 2-tuple with the new socket as the first element,
    and the port that was bound as the second. (Useful when passing 0 as a port
    number to bind any free port.)

    Returns:
        (socket, port)

    """
    ss = netutil.bind_sockets(port=port or 0, address=address)
    assert len(ss)
    ports = {s.getsockname()[1] for s in ss}
    assert len(ports) == 1, 'Multiple ports assigned??'
    actual_port = ports.pop()
    if port:
        assert actual_port == port
    return (ss, actual_port)