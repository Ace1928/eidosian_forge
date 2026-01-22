import numpy as np
import sys
from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets
import pyaudio
def set_parameter(self, parameter: str, value: float):
    if parameter in ['attack', 'decay', 'sustain', 'release']:
        setattr(self, parameter, value)