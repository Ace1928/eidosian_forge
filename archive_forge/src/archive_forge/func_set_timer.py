from deprecated.sphinx import deprecated
import contextlib
import logging
import time
import sys
def set_timer(self, timer='wall'):
    """Set the timer function

        Parameters
        ----------
        timer : {'wall', 'cpu', or callable}
                Timer function used to measure task running times.
                'wall' uses `time.time`, 'cpu' uses `time.process_time`

        Returns
        -------
        self
        """
    if timer == 'wall':
        timer = time.time
    elif timer == 'cpu':
        timer = time.process_time
    elif not callable(timer):
        raise ValueError("Expected timer to be 'wall', 'cpu', or a callable. Got {}".format(timer))
    self.timer = timer
    return self