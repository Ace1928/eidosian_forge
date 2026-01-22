from __future__ import absolute_import, division, print_function
import sys
import numpy as np
from ..audio.signal import signal_frame, smooth as smooth_signal
from ..ml.nn import average_predictions
from ..processors import (OnlineProcessor, ParallelProcessor, Processor,
def process_online(self, activations, reset=True, **kwargs):
    """
        Detect the beats in the given activation function with the forward
        algorithm.

        Parameters
        ----------
        activations : numpy array
            Beat activation for a single frame.
        reset : bool, optional
            Reset the DBNBeatTrackingProcessor to its initial state before
            processing.

        Returns
        -------
        beats : numpy array
            Detected beat position [seconds].

        """
    if reset:
        self.reset()
    fwd = self.hmm.forward(activations, reset=reset)
    states = np.argmax(fwd, axis=1)
    beats = self.om.pointers[states] == 1
    positions = self.st.state_positions[states]
    if self.visualize and len(activations) == 1:
        beat_length = 80
        display = [' '] * beat_length
        display[int(positions * beat_length)] = '*'
        strength_length = 10
        self.strength = int(max(self.strength, activations * 10))
        display.append('| ')
        display.extend(['*'] * self.strength)
        display.extend([' '] * (strength_length - self.strength))
        if self.counter % 5 == 0:
            self.strength -= 1
        if beats:
            self.beat_counter = 3
        if self.beat_counter > 0:
            display.append('| X ')
        else:
            display.append('|   ')
        self.beat_counter -= 1
        display.append('| %5.1f | ' % self.tempo)
        sys.stderr.write('\r%s' % ''.join(display))
        sys.stderr.flush()
    beats_ = []
    for frame in np.nonzero(beats)[0]:
        cur_beat = (frame + self.counter) / float(self.fps)
        next_beat = self.last_beat + 60.0 / self.max_bpm
        if cur_beat >= next_beat:
            self.tempo = 60.0 / (cur_beat - self.last_beat)
            self.last_beat = cur_beat
            beats_.append(cur_beat)
    self.counter += len(activations)
    return np.array(beats_)