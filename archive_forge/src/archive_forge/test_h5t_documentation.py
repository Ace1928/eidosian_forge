import numpy as np
import h5py
from h5py import h5t
from .common import TestCase, ut
Custom floats are correctly promoted to standard floats on read.