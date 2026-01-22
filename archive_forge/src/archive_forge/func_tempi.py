from __future__ import absolute_import, division, print_function
import numpy as np
import mido
@property
def tempi(self):
    """
        Tempi (mircoseconds per quarter note) of the MIDI file.

        Returns
        -------
        tempi : numpy array
            Array with tempi (time, tempo).

        Notes
        -----
        The time will be given in the unit set by `unit`.

        """
    tempi = []
    for msg in self:
        if msg.type == 'set_tempo':
            tempi.append((msg.time, msg.tempo))
    if not tempi or tempi[0][0] > 0:
        tempi.insert(0, (0, DEFAULT_TEMPO))
    return np.asarray(tempi, np.float)