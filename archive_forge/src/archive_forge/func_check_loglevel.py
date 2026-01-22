import enum
import logging
import logging.handlers
import asyncio
import socket
import queue
import argparse
import yaml
from . import defaults
def check_loglevel(arg):
    try:
        return LogLevel[arg]
    except (IndexError, KeyError):
        raise argparse.ArgumentTypeError('%s is not valid loglevel' % (repr(arg),))