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
def reopen_files(self):
    if self.cfg.capture_output and self.cfg.errorlog != '-':
        for stream in (sys.stdout, sys.stderr):
            stream.flush()
        with self.lock:
            if self.logfile is not None:
                self.logfile.close()
            self.logfile = open(self.cfg.errorlog, 'a+')
            os.dup2(self.logfile.fileno(), sys.stdout.fileno())
            os.dup2(self.logfile.fileno(), sys.stderr.fileno())
    for log in loggers():
        for handler in log.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.acquire()
                try:
                    if handler.stream:
                        handler.close()
                        handler.stream = handler._open()
                finally:
                    handler.release()