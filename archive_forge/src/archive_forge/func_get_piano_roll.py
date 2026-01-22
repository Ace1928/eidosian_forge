import numpy as np
import os
import pkg_resources
from .containers import PitchBend
from .utilities import pitch_bend_to_semitones, note_number_to_hz
def get_piano_roll(self, fs=100, times=None, pedal_threshold=64):
    """Compute a piano roll matrix of this instrument.

        Parameters
        ----------
        fs : int
            Sampling frequency of the columns, i.e. each column is spaced apart
            by ``1./fs`` seconds.
        times : np.ndarray
            Times of the start of each column in the piano roll.
            Default ``None`` which is ``np.arange(0, get_end_time(), 1./fs)``.
        pedal_threshold : int
            Value of control change 64 (sustain pedal) message that is less
            than this value is reflected as pedal-off.  Pedals will be
            reflected as elongation of notes in the piano roll.
            If None, then CC64 message is ignored.
            Default is 64.

        Returns
        -------
        piano_roll : np.ndarray, shape=(128,times.shape[0])
            Piano roll of this instrument.

        """
    if self.notes == []:
        return np.array([[]] * 128)
    end_time = self.get_end_time()
    if times is not None and times[-1] > end_time:
        end_time = times[-1]
    piano_roll = np.zeros((128, int(fs * end_time)))
    if self.is_drum:
        if times is None:
            return piano_roll
        else:
            return np.zeros((128, times.shape[0]))
    for note in self.notes:
        piano_roll[note.pitch, int(note.start * fs):int(note.end * fs)] += note.velocity
    if pedal_threshold is not None:
        CC_SUSTAIN_PEDAL = 64
        time_pedal_on = 0
        is_pedal_on = False
        for cc in [_e for _e in self.control_changes if _e.number == CC_SUSTAIN_PEDAL]:
            time_now = int(cc.time * fs)
            is_current_pedal_on = cc.value >= pedal_threshold
            if not is_pedal_on and is_current_pedal_on:
                time_pedal_on = time_now
                is_pedal_on = True
            elif is_pedal_on and (not is_current_pedal_on):
                subpr = piano_roll[:, time_pedal_on:time_now]
                pedaled = np.maximum.accumulate(subpr, axis=1)
                piano_roll[:, time_pedal_on:time_now] = pedaled
                is_pedal_on = False
    ordered_bends = sorted(self.pitch_bends, key=lambda bend: bend.time)
    end_bend = PitchBend(0, end_time)
    for start_bend, end_bend in zip(ordered_bends, ordered_bends[1:] + [end_bend]):
        if np.abs(start_bend.pitch) < 1:
            continue
        start_pitch = pitch_bend_to_semitones(start_bend.pitch)
        bend_int = int(np.sign(start_pitch) * np.floor(np.abs(start_pitch)))
        bend_decimal = np.abs(start_pitch - bend_int)
        bend_range = np.r_[int(start_bend.time * fs):int(end_bend.time * fs)]
        bent_roll = np.zeros(piano_roll[:, bend_range].shape)
        if start_bend.pitch >= 0:
            if bend_int != 0:
                bent_roll[bend_int:] = piano_roll[:-bend_int, bend_range]
            else:
                bent_roll = piano_roll[:, bend_range]
            bent_roll[1:] = (1 - bend_decimal) * bent_roll[1:] + bend_decimal * bent_roll[:-1]
        else:
            if bend_int != 0:
                bent_roll[:bend_int] = piano_roll[-bend_int:, bend_range]
            else:
                bent_roll = piano_roll[:, bend_range]
            bent_roll[:-1] = (1 - bend_decimal) * bent_roll[:-1] + bend_decimal * bent_roll[1:]
        piano_roll[:, bend_range] = bent_roll
    if times is None:
        return piano_roll
    piano_roll_integrated = np.zeros((128, times.shape[0]))
    times = np.array(np.round(times * fs), dtype=np.int32)
    for n, (start, end) in enumerate(zip(times[:-1], times[1:])):
        if start < piano_roll.shape[1]:
            if start == end:
                end = start + 1
            piano_roll_integrated[:, n] = np.mean(piano_roll[:, start:end], axis=1)
    return piano_roll_integrated