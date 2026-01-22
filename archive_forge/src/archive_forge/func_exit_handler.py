import os
import argparse
import asyncio
import grp
import logging
import pwd
import signal
import sys
from functools import partial
from .asdnotify import AsyncSystemdNotifier
from . import utils
from . import defaults
from .proactive_fetcher import STSProactiveFetcher
from .responder import STSSocketmapResponder
def exit_handler(exit_event, signum, frame):
    logger = logging.getLogger('MAIN')
    if exit_event.is_set():
        logger.warning('Got second exit signal! Terminating hard.')
        os._exit(1)
    else:
        logger.warning('Got first exit signal! Terminating gracefully.')
        exit_event.set()