from __future__ import annotations
import re
from typing import Match, cast
from zmq.backend import zmq_version_info
def pyzmq_version_info() -> tuple[int, int, int] | tuple[int, int, int, float]:
    """return the pyzmq version as a tuple of at least three numbers

    If pyzmq is a development version, `inf` will be appended after the third integer.
    """
    return version_info