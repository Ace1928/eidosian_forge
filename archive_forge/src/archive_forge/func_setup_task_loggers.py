import logging
import os
import sys
import warnings
from logging.handlers import WatchedFileHandler
from kombu.utils.encoding import set_default_encoding_file
from celery import signals
from celery._state import get_current_task
from celery.exceptions import CDeprecationWarning, CPendingDeprecationWarning
from celery.local import class_property
from celery.utils.log import (ColorFormatter, LoggingProxy, get_logger, get_multiprocessing_logger, mlevel,
from celery.utils.nodenames import node_format
from celery.utils.term import colored
def setup_task_loggers(self, loglevel=None, logfile=None, format=None, colorize=None, propagate=False, **kwargs):
    """Setup the task logger.

        If `logfile` is not specified, then `sys.stderr` is used.

        Will return the base task logger object.
        """
    loglevel = mlevel(loglevel or self.loglevel)
    format = format or self.task_format
    colorize = self.supports_color(colorize, logfile)
    logger = self.setup_handlers(get_logger('celery.task'), logfile, format, colorize, formatter=TaskFormatter, **kwargs)
    logger.setLevel(loglevel)
    logger.propagate = int(propagate)
    signals.after_setup_task_logger.send(sender=None, logger=logger, loglevel=loglevel, logfile=logfile, format=format, colorize=colorize)
    return logger