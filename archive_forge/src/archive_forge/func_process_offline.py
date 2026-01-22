from __future__ import absolute_import, division, print_function
import sys
import numpy as np
from ..audio.signal import signal_frame, smooth as smooth_signal
from ..ml.nn import average_predictions
from ..processors import (OnlineProcessor, ParallelProcessor, Processor,
def process_offline(self, activations, **kwargs):
    """
        Detect the beats in the given activation function with Viterbi
        decoding.

        Parameters
        ----------
        activations : numpy array
            Beat activation function.

        Returns
        -------
        beats : numpy array
            Detected beat positions [seconds].

        """
    beats = np.empty(0, dtype=np.int)
    first = 0
    if self.threshold:
        idx = np.nonzero(activations >= self.threshold)[0]
        if idx.any():
            first = max(first, np.min(idx))
            last = min(len(activations), np.max(idx) + 1)
        else:
            last = first
        activations = activations[first:last]
    if not activations.any():
        return beats
    path, _ = self.hmm.viterbi(activations)
    if self.correct:
        beat_range = self.om.pointers[path]
        idx = np.nonzero(np.diff(beat_range))[0] + 1
        if beat_range[0]:
            idx = np.r_[0, idx]
        if beat_range[-1]:
            idx = np.r_[idx, beat_range.size]
        if idx.any():
            for left, right in idx.reshape((-1, 2)):
                peak = np.argmax(activations[left:right]) + left
                beats = np.hstack((beats, peak))
    else:
        from scipy.signal import argrelmin
        beats = argrelmin(self.st.state_positions[path], mode='wrap')[0]
        beats = beats[self.om.pointers[path[beats]] == 1]
    return (beats + first) / float(self.fps)