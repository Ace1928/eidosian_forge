import numpy as np
import sys
from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets
import pyaudio
def phase_modulation(self, carrier: np.ndarray, modulator: np.ndarray, index: float) -> np.ndarray:
    """Performs phase modulation on a sound."""
    return np.sin(2 * np.pi * carrier + index * modulator)