from __future__ import absolute_import, division, print_function
import numpy as np
import mido
def note_hash(channel, pitch):
    """Generate a note hash."""
    return channel * 128 + pitch