import time
import threading
@property
def current_rate(self):
    """The current transfer rate

        :rtype: float
        :returns: The current tracked transfer rate
        """
    if self._last_time is None:
        return 0.0
    return self._current_rate