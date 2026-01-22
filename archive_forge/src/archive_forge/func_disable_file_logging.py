import logging
from warnings import warn
import os
import sys
from .misc import str2bool
def disable_file_logging(self):
    if self._hdlr:
        self._logger.removeHandler(self._hdlr)
        self._utlogger.removeHandler(self._hdlr)
        self._iflogger.removeHandler(self._hdlr)
        self._fmlogger.removeHandler(self._hdlr)
        self._hdlr = None