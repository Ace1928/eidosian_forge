import numpy as np
import sys
from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets
import pyaudio
def frequency_modulation(self, carrier: np.ndarray, modulator: np.ndarray, index: float) -> np.ndarray:
    """Performs frequency modulation on a sound."""
    return np.sin(2 * np.pi * carrier * np.cumsum(1 + index * modulator))