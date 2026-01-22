import gc
import os
import warnings
import threading
import contextlib
from abc import ABCMeta, abstractmethod
from ._multiprocessing_helpers import mp
def batch_completed(self, batch_size, duration):
    """Callback indicate how long it took to run a batch"""
    if batch_size == self._effective_batch_size:
        old_duration = self._smoothed_batch_duration
        if old_duration == self._DEFAULT_SMOOTHED_BATCH_DURATION:
            new_duration = duration
        else:
            new_duration = 0.8 * old_duration + 0.2 * duration
        self._smoothed_batch_duration = new_duration