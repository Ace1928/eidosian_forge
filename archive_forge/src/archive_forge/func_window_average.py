import bisect
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import DefaultDict, Dict, List, Optional
from ray.serve._private.constants import SERVE_LOGGER_NAME
def window_average(self, key: str, window_start_timestamp_s: float, do_compact: bool=True) -> Optional[float]:
    """Perform a window average operation for metric `key`

        Args:
            key: the metric name.
            window_start_timestamp_s: the unix epoch timestamp for the
              start of the window. The computed average will use all datapoints
              from this timestamp until now.
            do_compact: whether or not to delete the datapoints that's
              before `window_start_timestamp_s` to save memory. Default is
              true.
        Returns:
            The average of all the datapoints for the key on and after time
            window_start_timestamp_s, or None if there are no such points.
        """
    points_after_idx = self._get_datapoints(key, window_start_timestamp_s)
    if do_compact:
        self.data[key] = points_after_idx
    if len(points_after_idx) == 0:
        return
    return sum((point.value for point in points_after_idx)) / len(points_after_idx)