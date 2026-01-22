from __future__ import annotations
import numbers
import socket
import sys
from datetime import datetime
from signal import Signals
from types import FrameType
from typing import Any
from celery import VERSION_BANNER, Celery, beat, platforms
from celery.utils.imports import qualname
from celery.utils.log import LOG_LEVELS, get_logger
from celery.utils.time import humanize_seconds
def start_scheduler(self) -> None:
    if self.pidfile:
        platforms.create_pidlock(self.pidfile)
    service = self.Service(app=self.app, max_interval=self.max_interval, scheduler_cls=self.scheduler_cls, schedule_filename=self.schedule)
    if not self.quiet:
        print(self.banner(service))
    self.setup_logging()
    if self.socket_timeout:
        logger.debug('Setting default socket timeout to %r', self.socket_timeout)
        socket.setdefaulttimeout(self.socket_timeout)
    try:
        self.install_sync_handler(service)
        service.start()
    except Exception as exc:
        logger.critical('beat raised exception %s: %r', exc.__class__, exc, exc_info=True)
        raise