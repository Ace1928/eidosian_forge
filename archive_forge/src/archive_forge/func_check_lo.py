from __future__ import annotations
import numpy as np
from .channels import DriveChannel, MeasureChannel
from .exceptions import PulseError
def check_lo(self, channel: DriveChannel | MeasureChannel, freq: float) -> bool:
    """Check that lo is valid for channel.

        Args:
            channel: Channel to validate lo for
            freq: lo frequency
        Raises:
            PulseError: If freq is outside of channels range
        Returns:
            True if lo is valid for channel
        """
    lo_ranges = self._lo_ranges
    if channel in lo_ranges:
        lo_range = lo_ranges[channel]
        if not lo_range.includes(freq):
            raise PulseError(f'Specified LO freq {freq:f} is out of range {lo_range}')
    return True