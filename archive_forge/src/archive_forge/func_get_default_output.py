from abc import ABCMeta, abstractmethod
from enum import Enum, auto
from typing import Dict, Optional
from pyglet import event
@abstractmethod
def get_default_output(self):
    """Returns a default active output device or None if none available."""
    pass