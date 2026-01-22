import base64
import binascii
import json
import time
import logging
from logging.config import dictConfig
from logging.config import fileConfig
import os
import socket
import sys
import threading
import traceback
from gunicorn import util
def close_on_exec(self):
    for log in loggers():
        for handler in log.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.acquire()
                try:
                    if handler.stream:
                        util.close_on_exec(handler.stream.fileno())
                finally:
                    handler.release()