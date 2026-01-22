from __future__ import print_function
import mido
import numpy as np
import math
import warnings
import collections
import copy
import functools
import six
from heapq import merge
from .instrument import Instrument
from .containers import (KeySignature, TimeSignature, Lyric, Note,
from .utilities import (key_name_to_key_number, qpm_to_bpm)
def get_tempo_changes(self):
    """Return arrays of tempo changes in quarter notes-per-minute and their
        times.

        Returns
        -------
        tempo_change_times : np.ndarray
            Times, in seconds, where the tempo changes.
        tempi : np.ndarray
            What the tempo is, in quarter notes-per-minute, at each point in
            time in ``tempo_change_times``.

        """
    tempo_change_times = np.zeros(len(self._tick_scales))
    tempi = np.zeros(len(self._tick_scales))
    for n, (tick, tick_scale) in enumerate(self._tick_scales):
        tempo_change_times[n] = self.tick_to_time(tick)
        tempi[n] = 60.0 / (tick_scale * self.resolution)
    return (tempo_change_times, tempi)