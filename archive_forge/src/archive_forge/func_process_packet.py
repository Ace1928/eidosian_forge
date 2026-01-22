from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Iterator, Optional, Tuple, TypeVar, Union
import torch
import torio
from torch.utils._pytree import tree_map
def process_packet(self, timeout: Optional[float]=None, backoff: float=10.0) -> int:
    """Read the source media and process one packet.

        If a packet is read successfully, then the data in the packet will
        be decoded and passed to corresponding output stream processors.

        If the packet belongs to a source stream that is not connected to
        an output stream, then the data are discarded.

        When the source reaches EOF, then it triggers all the output stream
        processors to enter drain mode. All the output stream processors
        flush the pending frames.

        Args:
            timeout (float or None, optional): Timeout in milli seconds.

                This argument changes the retry behavior when it failed to
                process a packet due to the underlying media resource being
                temporarily unavailable.

                When using a media device such as a microphone, there are cases
                where the underlying buffer is not ready.
                Calling this function in such case would cause the system to report
                `EAGAIN (resource temporarily unavailable)`.

                * ``>=0``: Keep retrying until the given time passes.

                * ``0<``: Keep retrying forever.

                * ``None`` : No retrying and raise an exception immediately.

                Default: ``None``.

                Note:

                    The retry behavior is applicable only when the reason is the
                    unavailable resource. It is not invoked if the reason of failure is
                    other.

            backoff (float, optional): Time to wait before retrying in milli seconds.

                This option is effective only when `timeout` is effective. (not ``None``)

                When `timeout` is effective, this `backoff` controls how long the function
                should wait before retrying. Default: ``10.0``.

        Returns:
            int:
                ``0``
                A packet was processed properly. The caller can keep
                calling this function to buffer more frames.

                ``1``
                The streamer reached EOF. All the output stream processors
                flushed the pending frames. The caller should stop calling
                this method.
        """
    return self._be.process_packet(timeout, backoff)