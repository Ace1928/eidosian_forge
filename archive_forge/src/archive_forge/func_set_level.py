from deprecated.sphinx import deprecated
import contextlib
import logging
import time
import sys
def set_level(self, level=1):
    """Set the logging level

        Parameters
        ----------
        level : `int` or `bool` (optional, default: 1)
            If < -2, prints no messages.
            If False or >= -2, prints CRITICAL messages.
            If False or >= -1, prints ERROR messages.
            If False or >= 0, prints WARNING messages.
            If True or >= 1, prints INFO messages.
            If >= 2, prints all messages.

        Returns
        -------
        self
        """
    if isinstance(level, bool):
        if level:
            level = logging.INFO
            level_name = 'INFO'
        else:
            level = logging.WARNING
            level_name = 'WARNING'
    elif level >= 2:
        level = logging.DEBUG
        level_name = 'DEBUG'
    elif level >= 1:
        level = True
        return self.set_level(level)
    elif level >= 0:
        level = False
        return self.set_level(level)
    elif level >= -1:
        level = logging.ERROR
        level_name = 'ERROR'
    elif level >= -2:
        level = logging.CRITICAL
        level_name = 'CRITICAL'
    else:
        level = logging.IGNORE
        level_name = 'IGNORE'
    if not self.logger.handlers:
        self.logger.tasklogger = self
        self.logger.propagate = False
        handler = logging.StreamHandler(stream=self.stream)
        handler.setFormatter(logging.Formatter(fmt='%(message)s'))
        self.logger.addHandler(handler)
    if level != self.logger.level:
        self.level = level
        self.logger.setLevel(level)
        self.log_debug('Set {} logging to {}'.format(self.name, level_name))
    return self