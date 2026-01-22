from abc import ABCMeta, abstractmethod
from enum import Enum, auto
from typing import Dict, Optional
from pyglet import event
@abstractmethod
def get_input_devices(self):
    """Returns a list of all active input devices."""
    pass