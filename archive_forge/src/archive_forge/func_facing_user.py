from __future__ import absolute_import
import os
import logging
from urllib.request import urlopen
from traitlets import (
from ipywidgets import DOMWidget, Image, Video, Audio, register, widget_serialization
import ipywebrtc._version
import traitlets
@classmethod
def facing_user(cls, audio=True, **kwargs):
    """Convenience method to get the camera facing the user (often front)

        Parameters
        ----------
        audio: bool
            Capture audio or not
        **kwargs
            Extra keyword arguments passed to the `CameraStream`
        """
    return cls._facing(facing_mode='user', audio=audio, **kwargs)