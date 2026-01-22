from deprecated.sphinx import deprecated
import contextlib
import logging
import time
import sys
def log_warning(self, msg):
    """Log a WARNING message

        Convenience function to log a message to the default Logger

        Parameters
        ----------
        msg : str
            Message to be logged
        """
    self._log(self.logger.warning, msg)