from __future__ import annotations
import numpy as np
from .channels import DriveChannel, MeasureChannel
from .exceptions import PulseError
def channel_lo(self, channel: DriveChannel | MeasureChannel) -> float:
    """Return channel lo.

        Args:
            channel: Channel to get lo for
        Raises:
            PulseError: If channel is not configured
        Returns:
            Lo of supplied channel if present
        """
    if isinstance(channel, DriveChannel):
        if channel in self.qubit_los:
            return self.qubit_los[channel]
    if isinstance(channel, MeasureChannel):
        if channel in self.meas_los:
            return self.meas_los[channel]
    raise PulseError('Channel %s is not configured' % channel)