from collections import OrderedDict
import numpy as np
import os
import re
import struct
import sys
import time
import logging
def set_progress(self, progress=0, force=False):
    """set_progress(progress=0, force=False)

        Set the current progress. To avoid unnecessary progress updates
        this will only have a visual effect if the time since the last
        update is > 0.1 seconds, or if force is True.
        """
    self._progress = progress
    if not (force or time.time() - self._last_progress_update > 0.1):
        return
    self._last_progress_update = time.time()
    unit = self._unit or ''
    progressText = ''
    if unit == '%':
        progressText = '%2.1f%%' % progress
    elif self._max > 0:
        percent = 100 * float(progress) / self._max
        progressText = '%i/%i %s (%2.1f%%)' % (progress, self._max, unit, percent)
    elif progress > 0:
        if isinstance(progress, float):
            progressText = '%0.4g %s' % (progress, unit)
        else:
            progressText = '%i %s' % (progress, unit)
    self._update_progress(progressText)