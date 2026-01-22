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
def setup_logging_subsystem(self, loglevel=None, logfile=None, format=None, colorize=None, hostname=None, **kwargs):
    if self.already_setup:
        return
    if logfile and hostname:
        logfile = node_format(logfile, hostname)
    Logging._setup = True
    loglevel = mlevel(loglevel or self.loglevel)
    format = format or self.format
    colorize = self.supports_color(colorize, logfile)
    reset_multiprocessing_logger()
    receivers = signals.setup_logging.send(sender=None, loglevel=loglevel, logfile=logfile, format=format, colorize=colorize)
    if not receivers:
        root = logging.getLogger()
        if self.app.conf.worker_hijack_root_logger:
            root.handlers = []
            get_logger('celery').handlers = []
            get_logger('celery.task').handlers = []
            get_logger('celery.redirected').handlers = []
        self._configure_logger(root, logfile, loglevel, format, colorize, **kwargs)
        self._configure_logger(get_multiprocessing_logger(), logfile, loglevel if MP_LOG else logging.ERROR, format, colorize, **kwargs)
        signals.after_setup_logger.send(sender=None, logger=root, loglevel=loglevel, logfile=logfile, format=format, colorize=colorize)
        self.setup_task_loggers(loglevel, logfile, colorize=colorize)
    try:
        stream = logging.getLogger().handlers[0].stream
    except (AttributeError, IndexError):
        pass
    else:
        set_default_encoding_file(stream)
    logfile_name = logfile if isinstance(logfile, str) else ''
    os.environ.update(_MP_FORK_LOGLEVEL_=str(loglevel), _MP_FORK_LOGFILE_=logfile_name, _MP_FORK_LOGFORMAT_=format)
    return receivers