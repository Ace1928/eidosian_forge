import datetime
import logging
import logging.handlers
import os
import re
import socket
import sys
import threading
import ovs.dirs
import ovs.unixctl
import ovs.util
@staticmethod
def close_log_file():
    """Closes the current log file. (This is useful on Windows, to ensure
        that a reference to the file is not kept by the daemon in case of
        detach.)"""
    if Vlog.__log_file:
        logger = logging.getLogger('file')
        logger.removeHandler(Vlog.__file_handler)
        Vlog.__file_handler.close()