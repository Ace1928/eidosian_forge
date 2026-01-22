from os_ken import cfg
import inspect
import platform
import logging
import logging.config
import logging.handlers
import os
import sys
def early_init_log(level=None):
    global _EARLY_LOG_HANDLER
    _EARLY_LOG_HANDLER = logging.StreamHandler(sys.stderr)
    log = logging.getLogger()
    log.addHandler(_EARLY_LOG_HANDLER)
    if level is not None:
        log.setLevel(level)