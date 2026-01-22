import configparser
import logging
import logging.handlers
import os
import signal
import sys
from oslo_rootwrap import filters
from oslo_rootwrap import subprocess
def setup_syslog(execname, facility, level):
    try:
        handler = logging.handlers.SysLogHandler(address='/dev/log', facility=facility)
    except IOError:
        logging.warning('Unable to setup syslog, maybe /dev/log socket needs to be restarted. Ignoring syslog configuration options.')
        return
    rootwrap_logger = logging.getLogger()
    rootwrap_logger.setLevel(level)
    handler.setFormatter(logging.Formatter(os.path.basename(execname) + ': %(message)s'))
    rootwrap_logger.addHandler(handler)