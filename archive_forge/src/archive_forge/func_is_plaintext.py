import enum
import logging
import logging.handlers
import asyncio
import socket
import queue
import argparse
import yaml
from . import defaults
def is_plaintext(contenttype):
    return contenttype.lower().partition(';')[0].strip() == 'text/plain'