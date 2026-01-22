import sys
import os
import struct
import logging
import numpy as np
@property
def sampling(self):
    """The sampling (voxel distances) of the data (dz, dy, dx)."""
    return self._info['sampling']