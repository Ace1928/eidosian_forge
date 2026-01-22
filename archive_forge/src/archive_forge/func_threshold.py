import logging
import warnings
from ._log import log as _global_log
@threshold.setter
def threshold(self, level):
    self.setLevel(level)