from __future__ import absolute_import, division, print_function
import sys
import math
import struct
import numpy as np
import warnings
@velocity.setter
def velocity(self, velocity):
    """
        Set the velocity of the note event.

        Parameters
        ----------
        velocity : int
            Velocity of the note.

        """
    self.data[1] = velocity