from abc import ABCMeta, abstractmethod
from enum import Enum, auto
from typing import Dict, Optional
from pyglet import event
def on_device_removed(self, device: AudioDevice):
    """Event, occurs when an existing device is removed from the system."""
    pass