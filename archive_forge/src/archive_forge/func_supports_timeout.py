import gc
import os
import warnings
import threading
import contextlib
from abc import ABCMeta, abstractmethod
from ._multiprocessing_helpers import mp
@property
def supports_timeout(self):
    return self.supports_retrieve_callback